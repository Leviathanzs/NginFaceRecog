# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: grpcproto/infrandriver.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1cgrpcproto/infrandriver.proto\x12\x0c\x64riverinfran\"C\n\x11VerifyByIdRequest\x12\r\n\x05TrxID\x18\x01 \x01(\t\x12\x0f\n\x07ImgData\x18\x02 \x01(\t\x12\x0e\n\x06UserID\x18\x03 \x01(\t\"I\n\x14VerifyByImageRequest\x12\r\n\x05TrxID\x18\x01 \x01(\t\x12\x10\n\x08ImgData1\x18\x02 \x01(\t\x12\x10\n\x08ImgData2\x18\x03 \x01(\t\"m\n\x0eVerifyResponse\x12\r\n\x05TrxID\x18\x01 \x01(\t\x12\x0f\n\x07\x45rrCode\x18\x02 \x01(\t\x12\x0f\n\x07Message\x18\x03 \x01(\t\x12\x11\n\tTimestamp\x18\x04 \x01(\t\x12\x17\n\x0f\x43onfidenceScore\x18\x05 \x01(\x02\"\xc8\x01\n\x12IdentifyOneRequest\x12\r\n\x05TrxID\x18\x01 \x01(\t\x12\x10\n\x08TenantID\x18\x02 \x01(\t\x12\x11\n\tTimestamp\x18\x03 \x01(\t\x12\x10\n\x08\x44\x65viceID\x18\x04 \x01(\t\x12\x0f\n\x07ImgData\x18\x05 \x01(\t\x12\x14\n\x0c\x46\x61\x63\x65\x44\x65tector\x18\x06 \x01(\x05\x12\x13\n\x0b\x45mbeddingID\x18\x07 \x01(\t\x12\x18\n\x10\x45mbeddingFeature\x18\x08 \x03(\x02\x12\x16\n\x0e\x45mbeddingBytes\x18\t \x01(\x0c\"\xc1\x01\n\x13IdentifyOneResponse\x12\r\n\x05TrxID\x18\x01 \x01(\t\x12\x0e\n\x06Result\x18\x02 \x01(\t\x12\x0e\n\x06Status\x18\x03 \x01(\t\x12\x0f\n\x07\x45rrCode\x18\x04 \x01(\t\x12\x13\n\x0b\x45mbeddingID\x18\x05 \x01(\t\x12\x18\n\x10MatchEmbeddingID\x18\x06 \x01(\t\x12\x17\n\x0f\x43onfidenceScore\x18\x07 \x01(\x02\x12\x0f\n\x07Message\x18\x08 \x01(\t\x12\x11\n\tTimestamp\x18\t \x01(\t\"\x95\x01\n\x13IdentifyManyRequest\x12\r\n\x05TrxID\x18\x01 \x01(\t\x12\x10\n\x08TenantID\x18\x02 \x01(\t\x12\x11\n\tTimestamp\x18\x03 \x01(\t\x12\x10\n\x08\x44\x65viceID\x18\x04 \x01(\t\x12\x0f\n\x07ImgData\x18\x05 \x01(\t\x12\x14\n\x0c\x46\x61\x63\x65\x44\x65tector\x18\x06 \x01(\x05\x12\x11\n\tNumResult\x18\x07 \x01(\x05\"\xf8\x01\n\x14IdentifyManyResponse\x12\r\n\x05TrxID\x18\x01 \x01(\t\x12\x0e\n\x06Result\x18\x02 \x01(\t\x12\x0e\n\x06Status\x18\x03 \x01(\t\x12\x0f\n\x07\x45rrCode\x18\x04 \x01(\t\x12\x13\n\x0b\x45mbeddingID\x18\x05 \x01(\t\x12\x18\n\x10MatchEmbeddingID\x18\x06 \x03(\t\x12\x17\n\x0f\x43onfidenceScore\x18\x07 \x03(\x02\x12\x0f\n\x07Message\x18\x08 \x01(\t\x12\x11\n\tTimestamp\x18\t \x01(\t\x12\x34\n\x0eMatchingResult\x18\n \x03(\x0b\x32\x1c.driverinfran.IdentifyResult\"C\n\x0eIdentifyResult\x12\x18\n\x10MatchEmbeddingID\x18\x01 \x01(\t\x12\x17\n\x0f\x43onfidenceScore\x18\x02 \x01(\t\"\xd5\x01\n\x13RegisterUserRequest\x12\r\n\x05TrxID\x18\x01 \x01(\t\x12\x10\n\x08TenantID\x18\x02 \x01(\t\x12\x11\n\tTimestamp\x18\x03 \x01(\t\x12\x10\n\x08\x44\x65viceID\x18\x04 \x01(\t\x12\x0f\n\x07ImgData\x18\x05 \x01(\t\x12\x14\n\x0c\x46\x61\x63\x65\x44\x65tector\x18\x06 \x01(\x05\x12\x0c\n\x04Name\x18\x07 \x01(\t\x12\x10\n\x08Password\x18\x08 \x01(\t\x12\x0e\n\x06UserID\x18\t \x01(\t\x12\x13\n\x0b\x44\x65scription\x18\n \x01(\t\x12\x0c\n\x04rkey\x18\x0b \x01(\t\"\xa0\x01\n\x14RegisterUserResponse\x12\r\n\x05TrxID\x18\x01 \x01(\t\x12\x0e\n\x06Result\x18\x02 \x01(\t\x12\x0e\n\x06Status\x18\x03 \x01(\t\x12\x0f\n\x07\x45rrCode\x18\x04 \x01(\t\x12\x0e\n\x06UserID\x18\x05 \x01(\t\x12\x14\n\x0cUserUniqueId\x18\x06 \x01(\t\x12\x0f\n\x07Message\x18\x07 \x01(\t\x12\x11\n\tTimestamp\x18\x08 \x01(\t\"w\n\x11\x44\x65leteUserRequest\x12\r\n\x05TrxID\x18\x01 \x01(\t\x12\x10\n\x08TenantID\x18\x02 \x01(\t\x12\x11\n\tTimestamp\x18\x03 \x01(\t\x12\x10\n\x08\x44\x65viceID\x18\x04 \x01(\t\x12\x0e\n\x06UserID\x18\x05 \x01(\t\x12\x0c\n\x04rkey\x18\x06 \x01(\t\"\x86\x01\n\x12\x44\x65leteUserResponse\x12\r\n\x05TrxID\x18\x01 \x01(\t\x12\x0e\n\x06Result\x18\x02 \x01(\t\x12\x0e\n\x06Status\x18\x03 \x01(\t\x12\x0f\n\x07\x45rrCode\x18\x04 \x01(\t\x12\x0f\n\x07Message\x18\x05 \x01(\t\x12\x11\n\tTimestamp\x18\x06 \x01(\t\x12\x0c\n\x04rkey\x18\x07 \x01(\t2\x93\x04\n\x12\x44riverInfranFaceID\x12T\n\x0bIdentifyOne\x12 .driverinfran.IdentifyOneRequest\x1a!.driverinfran.IdentifyOneResponse\"\x00\x12W\n\x0cIdentifyMany\x12!.driverinfran.IdentifyManyRequest\x1a\".driverinfran.IdentifyManyResponse\"\x00\x12W\n\x0cRegisterUser\x12!.driverinfran.RegisterUserRequest\x1a\".driverinfran.RegisterUserResponse\"\x00\x12Q\n\nDeleteUser\x12\x1f.driverinfran.DeleteUserRequest\x1a .driverinfran.DeleteUserResponse\"\x00\x12M\n\nVerifyById\x12\x1f.driverinfran.VerifyByIdRequest\x1a\x1c.driverinfran.VerifyResponse\"\x00\x12S\n\rVerifyByImage\x12\".driverinfran.VerifyByImageRequest\x1a\x1c.driverinfran.VerifyResponse\"\x00\x42?\n\x1cio.infran.driverinfranfaceidB\x17\x44riverInfranFaceIDProtoP\x01\xa2\x02\x03RTGb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'grpcproto.infrandriver_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\034io.infran.driverinfranfaceidB\027DriverInfranFaceIDProtoP\001\242\002\003RTG'
  _globals['_VERIFYBYIDREQUEST']._serialized_start=46
  _globals['_VERIFYBYIDREQUEST']._serialized_end=113
  _globals['_VERIFYBYIMAGEREQUEST']._serialized_start=115
  _globals['_VERIFYBYIMAGEREQUEST']._serialized_end=188
  _globals['_VERIFYRESPONSE']._serialized_start=190
  _globals['_VERIFYRESPONSE']._serialized_end=299
  _globals['_IDENTIFYONEREQUEST']._serialized_start=302
  _globals['_IDENTIFYONEREQUEST']._serialized_end=502
  _globals['_IDENTIFYONERESPONSE']._serialized_start=505
  _globals['_IDENTIFYONERESPONSE']._serialized_end=698
  _globals['_IDENTIFYMANYREQUEST']._serialized_start=701
  _globals['_IDENTIFYMANYREQUEST']._serialized_end=850
  _globals['_IDENTIFYMANYRESPONSE']._serialized_start=853
  _globals['_IDENTIFYMANYRESPONSE']._serialized_end=1101
  _globals['_IDENTIFYRESULT']._serialized_start=1103
  _globals['_IDENTIFYRESULT']._serialized_end=1170
  _globals['_REGISTERUSERREQUEST']._serialized_start=1173
  _globals['_REGISTERUSERREQUEST']._serialized_end=1386
  _globals['_REGISTERUSERRESPONSE']._serialized_start=1389
  _globals['_REGISTERUSERRESPONSE']._serialized_end=1549
  _globals['_DELETEUSERREQUEST']._serialized_start=1551
  _globals['_DELETEUSERREQUEST']._serialized_end=1670
  _globals['_DELETEUSERRESPONSE']._serialized_start=1673
  _globals['_DELETEUSERRESPONSE']._serialized_end=1807
  _globals['_DRIVERINFRANFACEID']._serialized_start=1810
  _globals['_DRIVERINFRANFACEID']._serialized_end=2341
# @@protoc_insertion_point(module_scope)
