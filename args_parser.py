def parse(args):
    h, v, w, mu = False, False, False, False
    mu_val = 0.1
    args_list = args.split(" ")
    print(args_list)
    if "-h" in args_list:
        h = True
    if "-v" in args_list:
        v = True
    if "-w" in args_list:
        w = True
    if "-mu" in args_list:
        mu = True
    if mu:
        mu_val = float(args_list[args_list.index("-mu")+1])
    
    return h, v, w, mu, mu_val

