""" #problem1
#a.
print(chr(0)) #return " "
#b
x="\n"
print(repr(x),x)
#c
使用print时相当于自动进行了Unicode编码到字符的转换

#problem2
#a
UTF-8兼容ASCII码，对于英文字母十分友好
#b
test="hello中文"
utf8_encoded=test.encode("utf-8")
decode_utf8_bytes_to_str_wrong(utf8_encoded)
#c
t="\n"
t1=t.encode("utf-8")
print(t1.decode("utf-8")) """

