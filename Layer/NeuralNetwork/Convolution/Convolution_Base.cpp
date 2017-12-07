//======================================
// 畳み込みニューラルネットワークの結合レイヤー
//======================================
#include"stdafx.h"

#include"Convolution_FUNC.hpp"

#include"Convolution_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
Convolution_Base::Convolution_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, ::CreateRuntimeParameter(), i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
Convolution_Base::~Convolution_Base()
{
}