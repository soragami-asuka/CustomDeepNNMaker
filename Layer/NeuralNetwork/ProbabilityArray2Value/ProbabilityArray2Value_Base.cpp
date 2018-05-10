//======================================
// 出力信号分割レイヤー
//======================================
#include"stdafx.h"

#include"ProbabilityArray2Value_FUNC.hpp"

#include"ProbabilityArray2Value_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
ProbabilityArray2Value_Base::ProbabilityArray2Value_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
ProbabilityArray2Value_Base::~ProbabilityArray2Value_Base()
{
}
