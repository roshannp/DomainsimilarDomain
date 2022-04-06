import argparse
from difflib import SequenceMatcher 
import tldextract
from scipy.sparse import csr_matrix
import numpy as np
from dataprun import GenerateWL, GenerateDomain2IP
from numpy import array
import ngram
from statistics import mean

def charSimilar(s1, s2):
    ratio = SequenceMatcher(None, s1, s2).ratio()
    return ratio

def similarByName(d1, d2):

    d1_split = tldextract.extract(d1)
    d2_split = tldextract.extract(d2)

    domain_similarity = charSimilar(d1_split[1], d2_split[1])
    subdomain_similarity = charSimilar(d1_split[0], d2_split[0])
    #print(d1_split, d2_split)
    #print("Domain and Subdomain based Similarity:",domain_similarity, subdomain_similarity)

    return(domain_similarity*0.65 + subdomain_similarity*0.35)

''' $ Added Ngram to compare similarity between domain names'''
def Sim_ngram_value(d1, d2):
    return ngram.NGram.compare(domain1, domain2)

def domainSimilarityAlgorithm(domain1, domain2):
    """
    Determines the overall similarity between two domains
    Parameters:
        domain1 (string): the first domain
        domain2 (string): the second domain
    Returns:
        float: the similarity value
     """
    ''' $ Finding the mean of two similarity functions and returning their values'''
    text_similar = similarByName(domain1, domain2)
    ngram_similar = Sim_ngram_value(domain1, domain2)
    total_similar = round(mean([text_similar, ngram_similar]), 3)
    return total_similar

''' $ edited the csr matrix generation part '''
''' $ Generates the similarity matrix (S) '''
def comparedomainmass(domain_list):
    matrix_size = len(domain_list)
    idomain = {v: k for k, v in domain_list.items()}
    dense_list = [[0]*matrix_size]*matrix_size

    for i in range(matrix_size):
        for j in range(i,matrix_size):
            similarity = domainSimilarityAlgorithm(idomain[i],idomain[j])
            dense_list[j][i] = similarity

    dense_array = np.array(dense_list)
    csr_array = csr_matrix(dense_array)
    #print(csr_array)
    return csr_array

''' $ main function to execute the Domain Similarity Domain '''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputfile', type=str, nargs='+', required=True,
                        help="input DNS source file")
    FLAGS = parser.parse_args()
    filename= []
    filename= FLAGS.inputfile
    ''' $ Pruning the Log file and extraiting the domains drom the log file'''
    RL,DD,IPD = GenerateWL(filename)

    with open("domain.txt","w") as f:
        for i in DD:
            f.write("{} : {}\r\n".format(DD[i], i))
        f.close()

    result = comparedomainmass(DD)
    print(result)

    
if __name__ == '__main__':
    main()
    
    
