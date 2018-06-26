//======================================
// バッチ正規化レイヤー
//======================================
#include"stdafx.h"

#include"BatchExponentialNormalization_FUNC.hpp"

#include"BatchExponentialNormalization_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
BatchExponentialNormalization_Base::BatchExponentialNormalization_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, ::CreateRuntimeParameter(), i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
BatchExponentialNormalization_Base::~BatchExponentialNormalization_Base()
{
}

