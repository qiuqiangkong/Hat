'''
SUMMARY:  global variables
AUTHOR:   Qiuqiang Kong
Created:  2016.05.20
Modified: 2016.07.31 Fix bug of reset_id_to_zero()
--------------------------------------
'''
import backend as K

# record current id of layer
# Every layer has a unique id. 
_new_id = 0
def new_id():
    global _new_id
    curr_id = _new_id
    _new_id += 1
    return curr_id
    
# reset id to zero
def reset_id_to_zero():
    global _new_id
    _new_id = 0
