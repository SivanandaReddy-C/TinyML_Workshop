/**
  ******************************************************************************
  * @file    network_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-07-15T16:51:29+0530
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "network_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_network_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_network_weights_array_u64[29] = {
  0xbef83f8e3eae60d0U, 0xbe104d083e597198U, 0xbed5c7eebe1fc9a0U, 0xbe9f703e3e0dca88U,
  0x3eacf646bed513b4U, 0x3d6cf4103e34fe38U, 0x3dc8ded83eeab690U, 0xbefe70c6befba60aU,
  0x3cfd2c20bebd6d58U, 0x3ec4a102be8c1d5aU, 0xbe8ed3ea3d6f6c50U, 0x3d441dc0bec25b6eU,
  0xbe1a38883e8c7eeaU, 0x3d4538c0bec177a2U, 0x3e4bbb34bdb78618U, 0x3ef68dfe3dcfbff8U,
  0x3ea20548bc895e20U, 0xbe06e0143e6605a0U, 0x3eeefd623e9587a4U, 0xbeb0cbb83e5dd3dcU,
  0x3e5c5c6ebe03e733U, 0xbe3c85713e0016eeU, 0xbd63b4e0be424ea1U, 0xbb7344803e817d5bU,
  0x3e91b9973e44844aU, 0x3eaa1a153eb2bed3U, 0x3e8b5fadbe9ef9d6U, 0x3e7beb92be97d4a2U,
  0x3d6e73e83d2ff790U,
};


ai_handle g_network_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_network_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

