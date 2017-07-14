//======================================
// 活性関数レイヤー
//======================================
#include"stdafx.h"

#include"Pooling_FUNC.hpp"

#include"Pooling_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
Pooling_Base::Pooling_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
Pooling_Base::~Pooling_Base()
{
}

