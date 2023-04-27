import sys
import logging
import math
from tqdm.notebook import tqdm

class Logfn:
    def __init__(self, folder):
        self.folder = folder
    def __call__(self, message, tq=False):
        logging.basicConfig(filename=f'{self.folder}run.log',format='%(asctime)s - %(message)s', filemode='a+', level=logging.INFO)
        logging.info(message)
        if tq:
            tqdm.write(message)
        else:
            print(message)
            sys.stdout.flush()

def pretty_size(n,pow=0,b=1024,u='B',pre=['']+[p+'i'for p in'KMGTPEZY']):
    pow,n=min(int(math.log(max(n*b**pow,1),b)),len(pre)-1),n*b**pow
    return "%%.%if %%s%%s"%abs(pow%(-pow-1))%(n/b**float(pow),pre[pow],u)