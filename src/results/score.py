from rdkit import Chem
import numpy as np
from rdkit.Chem import Draw
from rdkit.Chem import AllChem,Descriptors

import pickle
import glob
import argparse
from collections import Counter

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
#         print('invalid',smiles)
        raise Exception("wrong")
        return smiles


def prepare_g(g):
    g = g[:-1]
    #g = np.array(g)
    g-=1

    return g.tolist()

def g2s(g):
    """
    convert the graph to smiles
    """
    #print(g)
    if not isinstance(g,list):
        g = g.tolist()
    #g = prepare_g(g)
    mol = Chem.RWMol()
    mol_idx = -1
    check_prev = 1
    for i in range(len(g)):
        if g[i]==509:
            break
        if g[i]<=472:
            check_prev = 1
            
            atomicnum = int(g[i])
            chiraltag = 0
            while atomicnum>118:
                chiraltag +=1
                atomicnum -= 118
           

            mol_idx+=1
            mol.AddAtom(Chem.Atom(atomicnum))
            #print(Chiraldic[chiraltag])
            mol.GetAtomWithIdx(mol_idx).SetChiralTag(Chiraldic[chiraltag])
                 
            

            
        elif g[i]<505:
            bondtype =g[i]- 118*4 
            #print(check_prev,mol_idx)
            end_indx = mol_idx-check_prev
            #print(mol_idx,end_indx)
            mol.AddBond(mol_idx,mol_idx-check_prev,Bonddic[bondtype])
            check_prev +=1
            
        elif g[i]>509:
            check_prev += g[i] - 509  
    smile = Chem.MolToSmiles(mol, isomericSmiles=False)
    smile = smile.replace('N(=O)O','N(=O)=O')
    smile = canonicalize_smiles(smile)
    return smile

# add arguments
def get_arguments():
    parser = argparse.ArgumentParser(description='g2gt')
    #file name, allow multiple file as input
    parser.add_argument('--file', type=str, nargs='+', help='file name')
    #beam size
    parser.add_argument('--beam', type=int, default=100, help='beam size')
    args = parser.parse_args()
    return args

Bonddic = {0:Chem.BondType.UNSPECIFIED,1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,
    4:Chem.BondType.QUADRUPLE,5:Chem.BondType.QUINTUPLE,6:Chem.BondType.HEXTUPLE,7:Chem.BondType.ONEANDAHALF,
    8:Chem.BondType.TWOANDAHALF,9:Chem.BondType.THREEANDAHALF,10:Chem.BondType.FOURANDAHALF,11:Chem.BondType.FIVEANDAHALF,12:Chem.BondType.AROMATIC,
    13:Chem.BondType.IONIC,14:Chem.BondType.HYDROGEN,15:Chem.BondType.THREECENTER,16:Chem.BondType.DATIVEONE,17:Chem.BondType.DATIVE,18:Chem.BondType.DATIVEL,19:Chem.BondType.DATIVER,20:Chem.BondType.OTHER,21:Chem.BondType.ZERO}

Chiraldic = {0:Chem.ChiralType.CHI_UNSPECIFIED,1:Chem.ChiralType.CHI_TETRAHEDRAL_CW,2:Chem.ChiralType.CHI_TETRAHEDRAL_CCW,3:Chem.ChiralType.CHI_OTHER}

# main function
if __name__ == "__main__":
    """
    This function is used to score the results. 
    """


    #args
    args = get_arguments()
    # load all files by loop
    files = []
    dics = []
    for file in args.file:
        files.append(file)
        tmp_dic = {}
        #regrex for read file name: out* or usptov*
        
        for name in glob.glob(file + '/out*') + glob.glob(file + '/usptov*'):
            print(name)
            f = open(name,"rb")
            while 1:
                try:
                    temp = pickle.load(f)
                    tmp_dic.update(temp)
                except EOFError:
                    break
            f.close()
        
        # print(len(tmp_dic))
        dics.append(tmp_dic)

    correct = 0

    #print(dic)
    total = len(dics[0])
    cur = 0
    beam = 100
    correct_ = 0
    topn=np.array([0 for i in range(beam)])
    topn2=np.array([0 for i in range(beam)])
    max_frag = False
    at_correct = 0
    cont=0 
    cont2 = 0
    out = open("g2gt-results.txt",'w')
    for i in dics[0]:
        cont +=1

        prod=""
    #     if i>=1000:
    #         break
        cur+=1

        topn_flag = [0 for i in range(beam)]
        topn_flag2 = [0 for i in range(beam)]

        g = dics[0][i][1][0:-1]
        try:
            gt = g2s(g)
            cont2+=1
        except:
            # print(cont,"need neutralize",g)
            gt = 'cc'
        gt_r = prod+">>"+gt

        all_smiles = []
        all_smiles_uniq = []
        for k in range(beam):
            
            ps = []
            preds = []
            for m in range(len(dics)):
                try:
                    p = dics[m][i][0][k][0][1:]
                except:
                    p=''
                ps.append(p)

    #         if np.array_equal(p,g) or np.array_equal(p2,g) or np.array_equal(p3,g):
    #             topn_flag[k] = 1

                try:
                    pred = g2s(p)
                    all_smiles.append(pred)
                except:
                    pred = "nan"
                preds.append(pred)


            for pred in preds:
                if pred==gt:
                    topn_flag[k] = 1
                    break

                if max_frag:
                    try:
                        pred_frags = get_maxfrag(pred)
                        gt_frags = get_maxfrag(gt)
                    except:
                        continue
                    
                    for pred in pred_frags:
                        for gt in gt_frags:
                            if pred == gt:
                                topn_flag[k] = 1

                else:
                    if pred==gt:
                        topn_flag[k] = 1

        
        out_count = Counter(all_smiles)
        
        topp = out_count.most_common(50)
        
        for j in range(len(topp)):
            if topp[j][0] == gt:
                topn_flag2[j] =1 

        
        for j in range(beam):
            if topn_flag[j]==1:
                #print(topn)
                topn[j:]+=1
                break
        for j in range(beam):
        
            if topn_flag2[j]==1:
                #print(topn)
                topn2[j:]+=1
                break
                
                


    # print(topn)
    # print(topn[0]/total)
    print(correct_/total)
    if max_frag:
        print("max_frag")

    if max_frag:
        print("max_frag")
    else:
        print('no_max_frag')
    # print("all correct\n",topn/cont)

    print('top 10\n',topn2[:10]/cont2)
    print('top 10,remove\n',topn2[:10]/cont)
    print(cont,cont2)
