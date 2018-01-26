# -*- coding: utf-8 -*-

import csv

#ending = ["skills", "skill"]

ending = ["ability", "abilities"]
bullet = ["+ ", "â €¢ ", "Â · ","Â · Â Â Â Â Â Â Â ","\. ", "\- ", "/ ","**,** ","** ","* ","á ","### ","‰ Û¢ ","å á å å å å å å å ","å á ","á "]
skills_dic={}

def startwith(term):
    for b in bullet:
        if term.startswith(b):
            term=term.replace(b, "")
            break
    return term

with open("NLX_b8384025-fa09-417d-ae10-96880fac86be_out_sort.csv") as f:
    freader = csv.reader(f)
    for row in freader:
        
        term = row[0]
                
        term_list = term.split()
        last = term_list[-1]
        frequency = row[1]
        
        #last word == ending
        if last.lower() in ending:
            
                new_term=startwith(term)
                
                if not new_term.lower() in skills_dic:
                    skills_dic.update({new_term.lower():frequency})
                else:                    
                    old = skills_dic[new_term.lower()]
                    new = int(old) + int(frequency)
                    skills_dic[new_term.lower()] = new
                   

with open('NLX_2011Q_skills_out_ability.csv', 'wb') as f:   
    w = csv.writer(f)
    for key, value in skills_dic.items():
        w.writerow([key, value])


               