status_code = int(input("显示代码："))
match status_code:
    case 101: description = "不及格"
    case 102: description = "及格"
    case 103: description = "差一点及格"
    case _: description = '未知代码 请联系管理员'
print('状态码描述：',description)