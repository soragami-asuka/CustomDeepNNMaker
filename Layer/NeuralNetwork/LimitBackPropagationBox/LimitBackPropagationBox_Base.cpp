//======================================
// 出力信号分割レイヤー
//======================================
#include"stdafx.h"

#include"LimitBackPropagationBox_FUNC.hpp"

#include"LimitBackPropagationBox_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
LimitBackPropagationBox_Base::LimitBackPropagationBox_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
LimitBackPropagationBox_Base::~LimitBackPropagationBox_Base()
{
}
