//======================================
// 出力信号分割レイヤー
//======================================
#include"stdafx.h"

#include"LimitBackPropagationRange_FUNC.hpp"

#include"LimitBackPropagationRange_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
LimitBackPropagationRange_Base::LimitBackPropagationRange_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
LimitBackPropagationRange_Base::~LimitBackPropagationRange_Base()
{
}
