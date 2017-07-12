//======================================
// 活性関数レイヤー
//======================================
#include"stdafx.h"

#include"MergeInput_FUNC.hpp"

#include"MergeInput_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
MergeInput_Base::MergeInput_Base(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNMult2SingleLayerBase(guid, ::CreateLearningSetting(), i_lpInputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
MergeInput_Base::~MergeInput_Base()
{
}



//===========================
// 固有関数
//===========================
