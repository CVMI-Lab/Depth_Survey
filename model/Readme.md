### Supported Methods
- [x] Marigold
- [x] UniDepth 


In order to integrate into our interface, we need to implement the prepare_input / prepare_output function.

The input of `prepare_input` is listed in [dataset part](../dataset/Readme.md).

The output of `prepare_output` should be a dict, containing 
- pred_depth : [H,W]