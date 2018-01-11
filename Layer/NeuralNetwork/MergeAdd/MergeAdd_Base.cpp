//======================================
// 活性関数レイヤー
//======================================
#include"stdafx.h"

#include"MergeAdd_FUNC.hpp"

#include"MergeAdd_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
MergeAdd_Base::MergeAdd_Base(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNMult2SingleLayerBase(guid, i_lpInputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
MergeAdd_Base::~MergeAdd_Base()
{
}

