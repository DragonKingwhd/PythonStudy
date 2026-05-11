for i in range(0,21):
    for k in range (0,34):
        z = 100-i-k
        if 5 * i + 3 * k + z/3 == 100 and z % 3 ==0 :
            print(f"{i},{k},{z}")