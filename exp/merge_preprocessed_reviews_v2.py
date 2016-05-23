
#!/usr/bin/env python


import variables_v2

filepath = variables_v2.DATAPATH

for sentiment in variables_v2.SENTIMENTS:
    print 'sentiment: ' + sentiment
    
    for category in variables_v2.CATEGORIES:
        print '\tcategory: ' + category

        with open(filepath + category + '/allno' + category + '_reviews_' + sentiment + '.txt', 'w') as outfile:        
            for subcategory in variables_v2.CATEGORIES:
                print '\tsubcategory: ' + subcategory
                if subcategory == category:
                    print '\tsubcategory == category, continue'
                    continue

                with open(filepath + category + '/reviews_' + sentiment + '.txt') as infile:
                    bunchsize = 600
                    bunch = []                
                    for onereview in infile:
                        bunch.extend(onereview)
                        if len(bunch) == bunchsize:
                            outfile.writelines(bunch)
                            bunch = []    
                    outfile.writelines(bunch)

        with open(filepath + category + '/allno' + category + '_ratings_' + sentiment + '.txt', 'w') as outfile:
            for subcategory in variables_v2.CATEGORIES:
                print '\tsubcategory: ' + subcategory
                if subcategory == category:
                    print '\tsubcategory == category, continue'
                    continue

                with open(filepath + category + '/ratings_' + sentiment + '.txt') as infile:
                    bunchsize = 5000
                    bunch = []                
                    for onereview in infile:
                        bunch.extend(onereview)
                        if len(bunch) == bunchsize:
                            outfile.writelines(bunch)
                            bunch = []
                    outfile.writelines(bunch)
