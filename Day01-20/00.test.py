Height= float(input("请输入你的身高（米）"))
Weight= float(input("请输入你的体重（千克）"))
BMI = Weight / (Height ** 2)
if BMI < 18.5:
    print("你的体重过轻")
elif 18.5 <= BMI < 24:
    print("你的体重正常")
elif 24 <= BMI < 28:
    print("你的体重过重")
else:
    print("你的体重肥胖")
print(f"你的bmi是:{BMI:.1f}")
# print(f'{bmi = :.1f}')
# print("你的BMI是：", BMI)