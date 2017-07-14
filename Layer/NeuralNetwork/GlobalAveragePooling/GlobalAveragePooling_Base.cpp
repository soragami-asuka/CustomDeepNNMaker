//======================================
// 活性関数レイヤー
//======================================
#include"stdafx.h"

#include"GlobalAveragePooling_FUNC.hpp"

#include"GlobalAveragePooling_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
GlobalAveragePooling_Base::GlobalAveragePooling_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
GlobalAveragePooling_Base::~GlobalAveragePooling_Base()
{
}
