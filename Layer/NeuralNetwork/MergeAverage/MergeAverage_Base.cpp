//======================================
// 活性関数レイヤー
//======================================
#include"stdafx.h"

#include"MergeAverage_FUNC.hpp"

#include"MergeAverage_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
MergeAverage_Base::MergeAverage_Base(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNMult2SingleLayerBase(guid, i_lpInputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
MergeAverage_Base::~MergeAverage_Base()
{
}

