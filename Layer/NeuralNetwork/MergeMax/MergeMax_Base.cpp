//======================================
// 活性関数レイヤー
//======================================
#include"stdafx.h"

#include"MergeMax_FUNC.hpp"

#include"MergeMax_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
MergeMax_Base::MergeMax_Base(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNMult2SingleLayerBase(guid, i_lpInputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
MergeMax_Base::~MergeMax_Base()
{
}

