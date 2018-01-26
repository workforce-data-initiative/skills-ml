import csv

matching=0

with open("NLX_2011Q_skills_out_bullet.csv") as f:
    freader = csv.reader(f)
    for row in freader:
        
        matching=0
        
        term = row[0].strip()
        
        term_list = term.split()
        last = term_list[-1]
        
        length = len(term_list)
        
        with open("stopwords.txt") as stop:
            for line in stop:
                stword = line.strip()
                
                if term.lower() == stword.lower():
                    matching=1
                    break
                else:
                    next
                        
            if matching == 0:
                print '\t'.join(row)
               