//======================================
// 活性関数レイヤー
//======================================
#include"stdafx.h"

#include"Dropout_FUNC.hpp"

#include"Dropout_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
Dropout_Base::Dropout_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, ::CreateRuntimeParameter(), i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
Dropout_Base::~Dropout_Base()
{
}
