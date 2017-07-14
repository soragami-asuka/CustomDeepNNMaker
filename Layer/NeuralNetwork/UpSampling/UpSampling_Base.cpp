//======================================
// 畳み込みニューラルネットワークの結合レイヤー
//======================================
#include"stdafx.h"

#include"UpSampling_FUNC.hpp"

#include"UpSampling_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
UpSampling_Base::UpSampling_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
UpSampling_Base::~UpSampling_Base()
{
}

