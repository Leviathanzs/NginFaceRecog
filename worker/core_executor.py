import io
import numpy as np
import logging

import sys
import os
from datetime import datetime
import time
import struct
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')

import infranlib.databases.redis_handler as IMQ
import infranlib.databases.pool_handler as WKP

emb_dict = None
device = None
backbone = None
transform = None
conn = None
detector = None
reference = None
input_size = None
crop_size = None
pnet = None
rnet = None
onet = None
embedding_size = 512
input_size=[96, 112]
num_embeddings_executor = 1000
emb_array = []
id_emb_array = []
first_embedding_number_gb = 0
last_embedding_number = 1000
logging.getLogger().setLevel(logging.INFO)
def loadAllLocalModel(executor_name, first_embedding_number, last_embedding_number):
    logging.info("Load All Embedding")
    global first_embedding_number_gb
    global last_embedding_number_gb
    first_embedding_number_gb = first_embedding_number
    last_embedding_number_gb = last_embedding_number
    logging.info("Loading Model to cache")
    global emb_dict
    global emb_array
    global id_emb_array

    embedings_store = IMQ.RedisHandler(db=2)

    if os.path.exists(f"embedding_id_{executor_name}.txt"):
        print("File exists.")
        id_emb_array_tmp = []
        id_emb_from_file = []
        # id_emb_array = []
        emb_array_tmp = []

        with open(f"embedding_id_{executor_name}.txt", "r") as file:
            id_emb_from_file = [line.strip() for line in file.readlines()]  # Removes extra newlines
        
        for embedding_key in id_emb_from_file:
            # logging.info("Embedding {} ".format(embedding_key))
            if embedding_key == '00000000-0000-0000-0000-000000000000#00000000-0000-0000-0000-000000000000':
                emb_array_tmp = [[0 for i in range(512)]]
                id_emb_array_tmp.append(embedding_key)
            else :
                embeddings = embedings_store.get(embedding_key)
                embeddings = np.load(io.BytesIO(embeddings), allow_pickle=True)
                for embedding in embeddings:
                    emb_array_tmp.append(embedding)
                    id_emb_array_tmp.append(embedding_key)
            # logging.info("Embedding {} ".format(embedding_key))

        emb_array = np.array(emb_array_tmp)
        id_emb_array = id_emb_array_tmp
        # logging.info("Embedding {} ".format(embedding_key))
    else:
        print("File does not exist.")
        # id_emb_array = []
        # emb_array = []

        id_emb_array_tmp = []
        # id_emb_array = []
        emb_array_tmp = []

        embedding_keys = embedings_store.query("KEYS *")
        embedding_keys = [ebk.decode('utf-8') for ebk in embedding_keys]
        # print(id_emb_array)
        # logging.info("Embedding Last {} ".format(last_embedding_number))
        # logging.info("Embedding {} ".format(embedding_keys))
        for embedding_key in embedding_keys[first_embedding_number:last_embedding_number]:
            # logging.info("Embedding {} ".format(embedding_key))
            embeddings = embedings_store.get(embedding_key)
            embeddings = np.load(io.BytesIO(embeddings), allow_pickle=True)
            # logging.info("Embedding {} ".format(embeddings))
            for embedding in embeddings:
                emb_array_tmp.append(embedding)
                id_emb_array_tmp.append(embedding_key)
        if len(emb_array_tmp) == 0:
            emb_array_tmp = [[0 for i in range(512)]]
            id_emb_array_tmp = ['00000000-0000-0000-0000-000000000000#00000000-0000-0000-0000-000000000000']

        emb_array = np.array(emb_array_tmp)
        id_emb_array = id_emb_array_tmp
        
        with open(f"embedding_id_{executor_name}.txt", "w") as file:
            file.write("\n".join(id_emb_array) + "\n")  # Ensures each UUID is on a new line

    logging.info("Local Predefind embedding size : {}".format(len(emb_array)))

def identification_dot_product(executor_name, embedding_id, embedding_feature, embedding_bytes, embedding_string, num_result=1, embedding_get=1):
    err_code = 0
    similarity = []
    result = np.array([])  
    id_emb_idx = []  

    try:
        global emb_array
        global id_emb_array

        if embedding_get == 0:
            r_proc_feature = IMQ.RedisHandler(db=1)
            embedding_feature_byte = r_proc_feature.get(embedding_id)
            embedding_feature = np.array(struct.unpack('512f', embedding_feature_byte))
            similarity = cosine_similarity(embedding_feature.reshape(1, -1), emb_array)

        elif embedding_get == 1:
            embedding_feature = np.array(embedding_feature).squeeze()
            if embedding_feature.ndim == 1:
                embedding_feature = embedding_feature.reshape(1, -1)
            
            logging.info(f"✅ Bentuk embedding_feature setelah perbaikan: {embedding_feature.shape}")
            similarity = cosine_similarity(embedding_feature, emb_array)

        elif embedding_get == 2:
            embedding_feature = np.load(io.BytesIO(embedding_bytes), allow_pickle=True).squeeze()
            if embedding_feature.ndim == 1:
                embedding_feature = embedding_feature.reshape(1, -1)
                
            similarity = cosine_similarity(embedding_feature, emb_array)

        similarity = list(similarity[0].clip(min=0, max=1))

        # Sort Similarity
        result = np.array(similarity)
        result[::-1].sort()
        if num_result > len(similarity):
            num_result = len(similarity)

        for res in result[:num_result]:
            idx = np.where(similarity == res)[0][0]
            id_emb_idx.append(idx)
            similarity = np.delete(similarity, idx, axis=0)

    except Exception as err:
        logging.error(f"❌ Error in identification_dot_product: {err}")
        err_code = 1

    return result[:num_result] if result.size else [0.0], \
           [id_emb_array[idx].split('#')[1] for idx in id_emb_idx] if id_emb_idx else [""], err_code

def register_feature(executor_name, embedding_id):
    global emb_array
    global id_emb_array

    success = True

    try:
        start_time = time.time()

        # emb_array.append(embedding_feature)
        # id_emb_array.append(embedding_id)
        logging.info("Executor {} - Shape Before Register {} in this executor".format(executor_name, emb_array.shape))
        embedings_store = IMQ.RedisHandler(db=2)
        embeddings = embedings_store.get(embedding_id)
        # print(embeddings)
        embeddings = np.load(io.BytesIO(embeddings), allow_pickle=True)
        emb_array = np.append(emb_array, embeddings, axis=0)
        for _ in embeddings:
            id_emb_array.append(embedding_id)
        logging.info("Executor {} - Shape After Register {} in this executor".format(executor_name, emb_array.shape))
        logging.info("Executor {} - ID After Register {} in this executor".format(executor_name, len(id_emb_array)))
        logging.info("{} - Register feature executed in time {}".format(executor_name, str(time.time()-start_time)))
        if (emb_array.shape[0]!= len(id_emb_array)):
            logging.error("{} - Error : Embedding and ID not same, Reload all embedding".format(executor_name))
            loadAllLocalModel(first_embedding_number_gb, last_embedding_number_gb)

        with open(f"embedding_id_{executor_name}.txt", "w") as file:
            file.write("\n".join(id_emb_array) + "\n")  # Ensures each UUID is on a new line

    except Exception as err:
        logging.error("{} - Error : {}".format(executor_name, err))
        success = False
    return "Done Register" if success else "Error Register"

def remove_feature(executor_name, embedding_ids):
    global emb_array
    global id_emb_array

    success = True

    try:
        # print(id_emb_array)
        start_time = time.time()

        for embedding_id in embedding_ids:
            logging.info("Executor {} - Try Remove {} in this executor".format(executor_name, embedding_id))
            try:
                idx = id_emb_array.index(embedding_id)
                logging.info("Executor {} - Remove Embedding {} in this executor".format(executor_name, embedding_id))
                logging.info("Executor {} - Shape Before Delete {} in this executor".format(executor_name, emb_array.shape))
                emb_array = emb_array.tolist()
                emb_array = np.delete(emb_array, idx, axis=0)
                # logging.info("Executor {} - After Delete2 {} in this executor".format(executor_name, emb_array2.shape))
                # emb_array.pop(idx)
                # emb_array = np.array(emb_array)
                logging.info("Executor {} - Shape After Delete {} in this executor".format(executor_name, emb_array.shape))
                # del emb_array[idx]
                del id_emb_array[idx]
                logging.info("Executor {} - ID After Delete {} in this executor".format(executor_name, len(id_emb_array)))
            except ValueError:
                logging.info("{} - {} not in this executor".format(executor_name, embedding_id))

        logging.info("{} - Remove feature executed in time {}".format(executor_name, str(time.time()-start_time)))

        with open(f"embedding_id_{executor_name}.txt", "w") as file:
            file.write("\n".join(id_emb_array) + "\n")  # Ensures each UUID is on a new line
    except Exception as err:
        logging.error("{} - Error : {}".format(executor_name, err))
        success = False
    return "Done Remove" if success else "Error Remove"

if __name__ == "__main__":
    redis = IMQ.RedisHandler(db=2)
    logging.info(redis.check_connection())
    # init_embeddings(dbh, redis)
    # loadAllLocalModel(dbh, redis,1)

    # data_embedding = IMQ.RedisHandler(db=2)

    # output = io.BytesIO()
    # list_embedding = data_embedding.query("KEYS *")
    # logging(len(list_embedding))
    # im = Image.open(os.path.dirname(os.path.abspath(__file__))+'/test.jpg')
    # im.save(output, format=im.format)

    # data_store.set('test-keys', output.getvalue())

    # embeddings = redis.get("dea4e044-17c7-4b9e-8e03-1e77fb5b9a50#e67a7f36-1142-4326-9669-7b9e48367169#31f16d91-85c4-4379-bdf3-4249b24d8527#127644.jpg")

    # local_embedding = np.load(io.BytesIO(embeddings), allow_pickle=True)

    # embedding_external = [0.07898702517155898, 0.9069239607844684, 0.5105271364811825, 0.40680543903472766, 0.29193705270501125, 0.8442523602043774, 0.1559666909935321, 0.3029247278462571, 0.10952505625027542, 0.7807120774146692, 0.5533379511849629, 0.4535272593545966, 0.7828276453849943, 0.21755829665152004, 0.9411024093293916, 0.37331083464446724, 0.22742056876689598, 0.4882657272725789, 0.36448885081378846, 0.5006538574549368, 0.06765217103304488, 0.33869602380605823, 0.04330501668814213, 0.16644302993965554, 0.7253396990066207, 0.6334054099891235, 0.9025931913128998, 0.571656553065466, 0.13488661830715998, 0.5866452121391009, 0.34261930580356115, 0.10916017927474109, 0.04728863310481801, 0.5966409842137755, 0.027501641014272216, 0.06166324969915071, 0.3574479840152509, 0.5002002605052565, 0.09630630061110645, 0.9111892962357321, 0.08913693367131414, 0.3847440802468941, 0.3824150539690162, 0.49097903232030116, 0.5811090279345824, 0.7387233812430593, 0.8486130015305627, 0.577437537380243, 0.04101861292961806, 0.18335229262224484, 0.8762894724465615, 0.5078050599175871, 0.28611180048444473, 0.6936023599361799, 0.5460063908023857, 0.7756120609198797, 0.42018217572014005, 0.634351717112265, 0.3664225489369306, 0.8254787804872583, 0.4015961560996021, 0.9218799870996514, 0.8739915854181166, 0.4026656039677129, 0.510105302048864, 0.7632110609681638, 0.3477794131267463, 0.3928320683307275, 0.9862999960615555, 0.6597610901443318, 0.5975404764670774, 0.10857842336918355, 0.08856363570992576, 0.21216965143729305, 0.3841052310814552, 0.43406979109019805, 0.16628168615517558, 0.3314083203927539, 0.07181215579849931, 0.9123808181208648, 0.7346887679601869, 0.49730353235302405, 0.6777002066313438, 0.9992971143261302, 0.29397314678209374, 0.3412491086971202, 0.0312222110816065, 0.11408183113462522, 0.8245079178728454, 0.032179107193489265, 0.9892755955445243, 0.19092261470033678, 0.9475661063426009, 0.12072653273971301, 0.24112371876812333, 0.7154238044414334, 0.9082772588615731, 0.683682048424109, 0.10366268115316535, 0.4606867419153776, 0.5342435648514087, 0.726805915887587, 0.3947638884701645, 0.6971525371458991, 0.06012673732049523, 0.5184256765567685, 0.9786344398843682, 0.04157148094968155, 0.8987241777015412, 0.6254517417440135, 0.16423676652198937, 0.7246693201521074, 0.4837127257874574, 0.5124985329144194, 0.1252497439879542, 0.7573710423167958, 0.15894659405052702, 0.7806733930053028, 0.6546333387661073, 0.6866980301616817, 0.33090814005616476, 0.1539220453489819, 0.1757827493301526, 0.5794049838227602, 0.08760157969347193, 0.2819986553968006, 0.5097953724762234, 0.461410364384758, 0.2299160955670725, 0.3831596702486799, 0.5552313915845178, 0.18977793898143536, 0.2751674900495241, 0.44555057357341155, 0.38686246080512354, 0.8574748351758614, 0.6110724261720417, 0.9769162673942576, 0.46878499050115063, 0.9675805881153203, 0.8431862719770431, 0.5981707205949575, 0.48807111560726624, 0.3099746838894365, 0.7056046396575835, 0.8407048710597097, 0.08236140546627613, 0.17059144729086084, 0.16087840763286854, 0.2993855595799756, 0.22304460555853023, 0.9146633686164691, 0.5494537342293896, 0.17769116682564634, 0.6673451834597989, 0.8963893465709055, 0.8590409620461139, 0.22363414960575523, 0.552150742073195, 0.9121105202745688, 0.5226529834650053, 0.9209289064146409, 0.6755526744845273, 0.7832428478722189, 0.5332094683541514, 0.02157946705228886, 0.011261292549178625, 0.9777573012856322, 0.9321379273116168, 0.7326789176062688, 0.32739532324834164, 0.23977617254792405, 0.2967387388211683, 0.8145509479375415, 0.3519841418624562, 0.5500746481493564, 0.8635834333272653, 0.16743334956490352, 0.28334299721509204, 0.6042448736942344, 0.48100172499671323, 0.29428519278863974, 0.2446858629870985, 0.02577310105088859, 0.4407858499984303, 0.9417855679606905, 0.49557666394507716, 0.9236375199890755, 0.21035827430151455, 0.5251350676712747, 0.7372189667337722, 0.0762375186228782, 0.038905629953288834, 0.03473550004813697, 0.982659427198018, 0.4525025330381903, 0.061643978428128854, 0.15657880952685788, 0.1231360329082446, 0.383391350461575, 0.9125164285863996, 0.9857662399331983, 0.6929784908928821, 0.9470282560176508, 0.09724133861794504, 0.8231736856893056, 0.8525952207584399, 0.4351559451052638, 0.522599605027564, 0.06110371021008365, 0.5607306580680508, 0.9777490943239807, 0.940100482108546, 0.48523185171080496, 0.21539721932937061, 0.8476233355784155, 0.7928782936479924, 0.923096333533329, 0.3700035416711569, 0.3156108617384512, 0.6340732266012621, 0.2563708281147883, 0.6022579230881784, 0.519668009971173, 0.6308811626122459, 0.641534365331271, 0.49044133108038146, 0.5941457568394275, 0.8539862773269877, 0.40287466843362607, 0.4667849289474083, 0.2676802584360597, 0.4221127414593383, 0.7321823653050061, 0.49454382404842157, 0.9120025945375216, 0.09346659022197135, 0.9462274130065814, 0.5850393123401529, 0.04339815476846132, 0.9320441575571008, 0.4418176383660307, 0.46283210828673926, 0.15226230001051577, 0.02077417895232092, 0.8624712968608051, 0.6242168478285892, 0.969621196038061, 0.7041398884869645, 0.7373102108193867, 0.6744757492528564, 0.8692985944075847, 0.5741768383758847, 0.9235936693733304, 0.7079397334780305, 0.6836412154121838, 0.03057600112032266, 0.07500629728711283, 0.5057770430315115, 0.866900938264, 0.7903112050977331, 0.2512008223301512, 0.14396049234348895, 0.742206809030198, 0.43781916847141356, 0.5813446191519324, 0.6365240346973657, 0.08580434591435226, 0.27276534508878747, 0.706826223084594, 0.5155623986504171, 0.35497078563741635, 0.4109278218374607, 0.23403561007710616, 0.5744317150408353, 0.4237776858108031, 0.09967300618353425, 0.7124624608799854, 0.03857704942811768, 0.3029001363815059, 0.48199193316997935, 0.35756404354713933, 0.8238475260212889, 0.24602621141233705, 0.6701592072901053, 0.6497296973901111, 0.8553226794842737, 0.05918079643417007, 0.8278167652770302, 0.7075010508486911, 0.7918586458262997, 0.6925877478444418, 0.5653237238568029, 0.12500702321296475, 0.7030296317494775, 0.9005028375027271, 0.7852460067544238, 0.006628079321721603, 0.5827245040361445, 0.8956614883192515, 0.1708516996590299, 0.9651040551389579, 0.3059476513163224, 0.5249427174747069, 0.8065760247919695, 0.09966455395519613, 0.4226790218708857, 0.4374297742237444, 0.7867253122088531, 0.5647296983136858, 0.24375897814369185, 0.4145603842515201, 0.6786602403440584, 0.7656570742475857, 0.8138957823363027, 0.8435670454174571, 0.0993990856961654, 0.01391664258995906, 0.7694428726702763, 0.8915616484113404, 0.5016861187121046, 0.04866964322408318, 0.38576469482720266, 0.45880271032213504, 0.7664155153744967, 0.6182961978289097, 0.8703124032685948, 0.5282354060500044, 0.6636080464387937, 0.9395043416083403, 0.31654614614104604, 0.6907286038299376, 0.6276128874632061, 0.5971499119628069, 0.20070645448693447, 0.12595536037160204, 0.8569233227984698, 0.4755856985084427, 0.13147461069217603, 0.4332187928229321, 0.26847606235912047, 0.9734502186146959, 0.45301340704857107, 0.9916431057068467, 0.7889789271844229, 0.47869543243096657, 0.9618297075608179, 0.12506128489833424, 0.44183816664837006, 0.5399354819736334, 0.24200372458592767, 0.08050516431116261, 0.5878452330152515, 0.7754969012298666, 0.7399085435451218, 0.2510739206589886, 0.30623967042289246, 0.7392333944892117, 0.7442796359864876, 0.5153157431453118, 0.9591025905748677, 0.8390445627267815, 0.3931060288018481, 0.48504100121618665, 0.05545586018214588, 0.955798694540988, 0.9022666948358152, 0.9223143422597849, 0.6696210234257273, 0.42285499499855916, 0.23820113022231437, 0.29780340005425365, 0.7915491655300879, 0.3215005751740083, 0.5086650268237953, 0.5114849957325652, 0.43227689581631623, 0.24348797931714894, 0.8573324306097164, 0.8586766360201016, 0.9089364583302845, 0.46223152447401616, 0.3486421660982023, 0.1326255119744404, 0.17789147209002643, 0.6008331507849836, 0.1920709374025935, 0.38798476720933506, 0.4650247198072204, 0.7536383887434233, 0.920890384763968, 0.2227133175639463, 0.3653078911128198, 0.8125273183038962, 0.32638288965264084, 0.09647927219094732, 0.28029557518234316, 0.8506682556823574, 0.2519733619687522, 0.4974591644029569, 0.3254800283508982, 0.5387068248845779, 0.4565834407703385, 0.39785033844026885, 0.340590012395818, 0.4894456668872941, 0.5170251188837471, 0.4558986715246113, 0.07837573619662908, 0.5493766534864689, 0.1587144817464291, 0.9246339149813059, 0.5700727813508115, 0.9989252642100791, 0.5557147223448164, 0.08004901804867481, 0.8442451795686683, 0.9404622341658411, 0.40011633675372604, 0.9094234536171371, 0.12171878231238553, 0.7189343639173714, 0.7592902767938137, 0.8159426232078396, 0.5092351431133209, 0.6143543633860887, 0.9432733506346016, 0.10337435378908622, 0.5560367210182061, 0.2904005712900194, 0.012067343721054558, 0.16638582006135683, 0.5706258571772416, 0.9254990696328337, 0.5696881690124876, 0.4485996280847748, 0.20075992968887568, 0.9634189464242874, 0.6053759369925928, 0.06596399587799429, 0.47819181991847226, 0.6681044681280011, 0.2131850203897515, 0.9008518589937248, 0.8607092157429417, 0.6539650797881174, 0.1512103485826457, 0.3291316893178884, 0.4478170567836537, 0.6254406894094141, 0.21492079804670716, 0.9023445746246043, 0.24983025143093984, 0.42261378047139264, 0.08193762438938501, 0.6006880297377418, 0.7995639193083713, 0.19629664138739766, 0.25829414556895913, 0.9499007333083245, 0.7838961714781842, 0.18929728533052503, 0.28312099279373804, 0.048848775824074875, 0.9253042327823582, 0.20589670762156842, 0.12869167249813385, 0.02970684250941913, 0.34933421174020507, 0.8202078425428889, 0.9846479048319469, 0.2481673904614674, 0.7109668798527904, 0.7667080312890185, 0.891017469052504, 0.35158171126119464, 0.715261652701741, 0.6165923221703812, 0.33812526415982036, 0.057134529626246655, 0.9337504973022343, 0.5412762554602294, 0.6603035031360416, 0.45998096160730106, 0.7669090076520526, 0.00684963147938733, 0.41592094485166264, 0.5434183486203018, 0.26235215851416305, 0.4826290128865126, 0.9131723537627934, 0.31894118264948235, 0.046116727356544485, 0.22834244383728597, 0.25841569991064695, 0.38186258848787547, 0.18258677448881055, 0.6003792092516548, 0.13501169037903982, 0.9731882618748506, 0.6273229246156649, 0.08156355007631366, 0.8791369417462181, 0.93533749317186, 0.7846409852064016, 0.8736348737612933, 0.2473649846548872, 0.12750555447935374, 0.5727859914027179, 0.6554286193273235, 0.014601374475535, 0.5325755820865078]

    # logging(embedding_external)
    # logging(list(local_embedding[0]))
