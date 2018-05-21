# Penn Treebank POS Tags
def tagMatching(postag):
    matching = 0
    if postag == "CC":      #CC	Coordinating conjunction
        return 1
    elif postag == "CD":    #CD	Cardinal number
        return 2
    elif postag == "DT":    #DT	Determiner
        return 3
    elif postag == "EX":    #EX	Existential there
        return 4
    elif postag == "FW":    #FW	Foreign word
        return 5
    elif postag == "IN":    #IN	Preposition or subordinating conjunction
        return 6
    elif postag == "JJ":    #JJ	Adjective
        return 7
    elif postag == "JJR":   #JJR	Adjective, comparative
        return 8
    elif postag == "JJS":    #JJS	Adjective, superlative
        return 9
    elif postag == "LS":    #LS	List item marker
        return 10
    elif postag == "MD":    #MD	Modal
        return 11
    elif postag == "NN":    #NN	Noun, singular or mass
        return 12
    elif postag == "NNS":    #NNS	Noun, plural
        return 13
    elif postag == "NNP":    #NNP	Proper noun, singular
        return 14
    elif postag == "NNPS":    #NNPS	Proper noun, plural
        return 15
    elif postag == "PDT":    #PDT	Predeterminer
        return 16    
    elif postag == "POS":    #POS	Possessive ending
        return 17
    elif postag == "PRP":    #PRP	Personal pronoun
        return 18
    elif postag == "PRP$":    #PRP$	Possessive pronoun
        return 19
    elif postag == "RB":    #RB	Adverb
        return 20
    elif postag == "RBR":    #RBR	Adverb, comparative
        return 21    
    elif postag == "RBS":    #RBS	Adverb, superlative
        return 22    
    elif postag == "RP":    #RP	Particle
        return 23    
    elif postag == "SYM":    #SYM	Symbol
        return 24    
    elif postag == "TO":    #TO	to
        return 25    
    elif postag == "UH":    #UH	Interjection
        return 26    
    elif postag == "VB":    #VB	Verb, base form
        return 27    
    elif postag == "VB":    #VB	VBD	Verb, past tense
        return 28    
    elif postag == "VBG":    #VBG	Verb, gerund or present participle
        return 29    
    elif postag == "VBN":    #VBN	Verb, past participle
        return 30    
    elif postag == "VBP":    #VBP	Verb, non-3rd person singular present
        return 31    
    elif postag == "VBZ":    #VBZ	Verb, 3rd person singular present
        return 32    
    elif postag == "WDT":    #WDT	Wh-determiner
        return 33    
    elif postag == "WP":    #WP	Wh-pronoun
        return 34    
    elif postag == "WP$":    #WP$	Possessive wh-pronoun
        return 35    
    elif postag == "WRB":    #WRB Wh-adverb
        return 36
    else:
        return 0
  
