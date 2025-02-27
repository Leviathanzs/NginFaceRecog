class Worker:
    workername = ""
    host = ""
    port = ""

    def toString(self):
        str1 = "workername: " + self.workername
        str1 = str1 + ", " + "host: " + self.host
        str1 = str1 + ", " + "port: " + self.port

        return str1
