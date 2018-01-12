//======================================
// 活性関数レイヤー
//======================================
#include"stdafx.h"

#include"MergeMultiply_FUNC.hpp"

#include"MergeMultiply_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
MergeMultiply_Base::MergeMultiply_Base(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNMult2SingleLayerBase(guid, i_lpInputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
MergeMultiply_Base::~MergeMultiply_Base()
{
}

