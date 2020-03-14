
import tensorflow as tf

def gunjan_activation(x):

    rt = 0
    
    if (x <= 0.125):
       rt = 0.000323252
    elif (x<0.25):
       rt = 0.002533084
    elif(x<0.375):
       rt = 0.008266825
    elif(x<0.5):
       rt = 0.01872696
    elif(x<0.625):
       rt = 0.034595316
    elif(x<0.75):
       rt = 0.635148952
    elif(x<0.875):
       rt = 0.082841509
    elif(x<1):
       rt = 0.114439958
    elif(x<1.1251):
       rt = 0.150136668
    elif(x<1.25):
       rt = 0.189161594
    elif(x<1.375):
       rt = 0.230757741
    elif(x<1.5):
       rt = 0.274229354
    elif(x<1.625):
       rt = 0.318966557
    elif(x<1.75):
       rt = 0.364452989
    elif(x<1.875):
       rt = 0.410262538
    elif(x<2):
       rt = 0.456050004
    elif(x<2.125):
       rt = 0.501539094
    elif(x<1.875):
       rt = 0.410262538
    elif(x<2):
       rt = 0.456050004
    elif(x<2.125):
       rt = 0.501539094
    elif(x<2.125):
       rt = 0.546509964
    elif(x<2.375):
       rt = 0.590787559
    elif(x<2.5):
       rt = 0.634231432
    elif(x<2.625):
       rt = 0.676727265
    elif(x<2.75):
       rt = 0.718180101
    elif(x<2.875):
       rt = 0.758509133
    elif(x<3):
       rt = 0.797643886
    elif(x<3.125):
       rt = 0.835521525
    elif(x<3.25):
       rt = 0.872085119
    elif(x<3.375):
       rt = 0.90728262
    elif(x<3.5):
       rt = 0.94106641
    elif(x<3.625):
       rt = 0.973393245
    elif (x<3.75):
       rt = 1.00422447
    else: 
       rt =0 
    print (rt)
    return (x)
gunjan_activation(1.65)