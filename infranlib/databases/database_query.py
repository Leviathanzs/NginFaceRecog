import psycopg2

def registerUser(dbh, user_name, username, password, description):
    rst = dbh.execute("INSERT INTO users (uname, username, upassword, description)" +
                    " VALUES (%s, %s, %s, %s) RETURNING uuid, uname, username, description;",
                    (user_name, username, password, description), fetch=True)
    return rst

def registerUserMember(dbh, user_uuid, user_member_name, user_member_username, user_member_password, user_member_unique_id, user_member_description):
    rst = dbh.execute("INSERT INTO user_member (user_uuid, mname, musername, mpassword, unique_id, description)" +
                    " VALUES (%s, %s, %s, %s, %s, %s) RETURNING uuid, user_uuid, unique_id, mname, description;",
                    (user_uuid, user_member_name, user_member_username, user_member_password, user_member_unique_id, user_member_description), fetch=True)
    return rst

def saveImage(dbh, user_uuid, user_member_uuid, file_path, memory_tag, drawing):
    rst = dbh.execute("INSERT into images (user_uuid, user_member_uuid, file_path, memory_tag, image_data)" +
                    " VALUES (%s, %s, %s, %s, %s) RETURNING uuid, user_member_uuid;",
                    (user_uuid, user_member_uuid, file_path, memory_tag , psycopg2.Binary(drawing)), fetch=True)
    return rst

def saveEmbedding(dbh, user_uuid, user_member_uuid, file_path, memory_tag, embedding):
    rst = dbh.execute("INSERT into embeddings (user_uuid, user_member_uuid, file_path, memory_tag, embedding_data)" +
                    " VALUES (%s, %s, %s, %s, %s) RETURNING uuid, user_uuid, user_member_uuid;",
                    (user_uuid, user_member_uuid, file_path, memory_tag , psycopg2.Binary(embedding)), fetch=True)
    return rst

def saveEmbedding(dbh, user_uuid, user_member_uuid, file_path, memory_tag, embedding, embedding_v):
    # Ensure embedding_v is a numpy array (or a list of floats)
    # embedding_v = np.array(embedding_v).tolist()  # Convert to list of floats if it's a numpy array

    # Modify the query to include embedding_v
    rst = dbh.execute("""
        INSERT INTO embeddings (user_uuid, user_member_uuid, file_path, memory_tag, embedding_data, embedding_v)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING uuid, user_uuid, user_member_uuid;
    """, (user_uuid, user_member_uuid, file_path, memory_tag, psycopg2.Binary(embedding), embedding_v), fetch=True)

    return rst

def loadImages(dbh, user_uuid):
    rst = dbh.fetchall("SELECT * FROM images WHERE user_uuid = '{0}' AND is_active = true".format(user_uuid))
    return rst

def loadEmbeddings(dbh, user_uuid):
    rst = dbh.fetchall("SELECT * FROM embeddings WHERE (user_uuid = '{0}' OR user_member_uuid = '{0}') AND is_active = true".format(user_uuid))
    return rst

def loadImageList(dbh, user_uuid, user_member_uuid=None):
    if user_member_uuid:
        rst = dbh.fetchall("SELECT uuid, file_path, memory_tag FROM images WHERE user_uuid = '{0}' AND user_member_uuid = '{1}' AND is_active = true".format(user_uuid, user_member_uuid))
    else:
        rst = dbh.fetchall("SELECT uuid, file_path, memory_tag FROM images WHERE user_uuid = '{0}' AND is_active = true".format(user_uuid))
    return rst

def loadEmbeddingList(dbh, user_uuid, user_member_uuid=None):
    if user_member_uuid:
        rst = dbh.fetchall("SELECT uuid, file_path, memory_tag, is_active FROM embeddings WHERE user_uuid = '{0}' AND user_member_uuid = '{1}' AND is_active = true".format(user_uuid, user_member_uuid))
    else:
        rst = dbh.fetchall("SELECT uuid, file_path, memory_tag, is_active FROM embeddings WHERE user_uuid = '{0}' AND is_active = true".format(user_uuid))
    return rst

def loadEmbedding(dbh, embedding_uuid):
    rst = dbh.fetchall("SELECT user_uuid, user_member_uuid, images_uuid, uuid, memory_tag, embedding_data FROM embeddings WHERE uuid = '{0}' AND is_active = true".format(embedding_uuid))
    return rst

def loadImage(dbh, image_uuid):
    rst = dbh.fetchall("SELECT user_uuid, user_member_uuid, uuid, memory_tag, file_path, image_data FROM images WHERE uuid = '{0}' AND is_active = true".format(image_uuid))
    return rst

def loadUserList(dbh):
    rst = dbh.fetchall("SELECT uuid FROM users WHERE is_active = true")
    return rst

def loadMemberUserList(dbh, user_uuid):
    rst = dbh.fetchall("SELECT uuid, unique_id FROM user_member WHERE user_uuid = '{0}' AND is_active = true".format(user_uuid))
    return rst

def loadMemberUser(dbh, user_uuid, user_member_uuid):
    rst = dbh.fetchall("SELECT uuid, user_uuid, unique_id, mname FROM user_member WHERE user_uuid = '{0}' AND uuid = '{1}' AND is_active = true".format(user_uuid, user_member_uuid))
    return rst

def deleteImage(dbh, image_uuid):
    rst = dbh.execute("UPDATE images SET is_active = false WHERE uuid = '{0}'".format(image_uuid))
    return rst

def deleteEmbedding(dbh, user_uuid, user_member_uuid, uuid):
    if user_member_uuid == uuid:
        rst = dbh.execute("UPDATE embeddings SET is_active = false WHERE user_uuid = '{0}' AND user_member_uuid = '{1}' RETURNING user_uuid, user_member_uuid, uuid".format(user_uuid, uuid))
    else:
        rst = dbh.execute("UPDATE embeddings SET is_active = false WHERE user_uuid = '{0}' AND user_member_uuid = '{1}' AND uuid = '{2}' RETURNING user_uuid, user_member_uuid, uuid".format(user_uuid, user_member_uuid, uuid))
    return rst

def setMessage(redis, key, message):
    redis.set(key, message)

def getMessage(redis, key):
    message = redis.get(key)
    return message

def getMessageList(redis, key):
    return redis.query(key+'*')

def deleteMessage(redis, key):
    redis.delete(key)

def save_embedding(dbh, user_uuid, user_member_uuid, memory_tag, embedding):
    rst = dbh.execute("INSERT into embeddings (user_uuid, user_member_uuid, memory_tag, embedding_data)" +
                    " VALUES (%s, %s, %s, %s) RETURNING uuid;", 
                    (user_uuid, user_member_uuid, memory_tag , psycopg2.Binary(embedding)), 
                    fetch=True)
