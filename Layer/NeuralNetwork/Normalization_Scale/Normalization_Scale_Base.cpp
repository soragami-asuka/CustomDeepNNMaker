//======================================
// バッチ正規化レイヤー
//======================================
#include"stdafx.h"

#include"Normalization_Scale_FUNC.hpp"

#include"Normalization_Scale_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
Normalization_Scale_Base::Normalization_Scale_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
Normalization_Scale_Base::~Normalization_Scale_Base()
{
}

