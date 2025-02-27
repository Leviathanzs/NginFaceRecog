import time
import base64
from multiprocessing import Process
from executor_infran_async import *
from read_worker_config import *
import infranlib.databases.redis_handler as IMQ
import uuid

worker_config_data = readWorkerConfigAsync()

# ✅ Membaca konfigurasi worker
worker_config_data = readWorkerConfigAsync()
driver_host = worker_config_data['host']
driver_port = worker_config_data['port']
worker_id = worker_config_data['worker_id']
executor_ids = worker_config_data['executors'].keys()

# ✅ Debugging untuk memastikan worker membaca konfigurasi dengan benar
print("=========================================")
print(f"🚀 Worker ID: {worker_id}")
print(f"🔌 Redis Host: {driver_host}")
print(f"🔌 Redis Port: {driver_port}")
print(f"⚡ Executors: {list(executor_ids)}")
print("=========================================")

process_list = []
worker_config_string = json.dumps(worker_config_data)

# ✅ Tes koneksi ke Redis
try:
    r_test = IMQ.RedisHandler(db=3)
    redis_status = r_test.check_connection()
    print(f"✅ Redis Ping: {redis_status}")  # Jika True, berarti koneksi ke Redis berhasil
except Exception as e:
    print(f"❌ Redis Connection Failed: {str(e)}")
    exit(1)  # Hentikan eksekusi jika Redis tidak bisa diakses

if __name__ == "__main__":
    index = 0
    for executor_id in executor_ids:
        # ✅ Debugging sebelum menjalankan executor
        print(f"🚀 Memulai executor: {executor_id} dengan UUID {uuid.uuid4()}")

        proc = Process(
            target=serve, 
            args=(
                driver_host, 
                worker_id, 
                executor_id, 
                str(uuid.uuid4()), 
                worker_config_data['executors'][executor_id]['first_embedding_number'], 
                worker_config_data['executors'][executor_id]['last_embedding_number']
            )
        )

        process_list.append(proc)
        proc.start()
        time.sleep(2)  # Memberi jeda agar tidak overload

        # ✅ Debugging setelah executor dijalankan
        print(f"✅ {executor_id} is running: {proc.is_alive()}")

        index += 1

    # ✅ Memastikan semua executor berjalan
    size = index
    for i in range(size):
        process_list[i].join()  # ✅ Diperbaiki dengan menambahkan ()