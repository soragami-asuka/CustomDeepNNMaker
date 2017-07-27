//======================================
// 活性関数レイヤー
//======================================
#include"stdafx.h"

#include"GaussianNoise_FUNC.hpp"

#include"GaussianNoise_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
GaussianNoise_Base::GaussianNoise_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, CreateRuntimeParameter(), i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
GaussianNoise_Base::~GaussianNoise_Base()
{
}

