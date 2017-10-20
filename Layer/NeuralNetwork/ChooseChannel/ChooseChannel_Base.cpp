//======================================
// 出力信号分割レイヤー
//======================================
#include"stdafx.h"

#include"ChooseChannel_FUNC.hpp"

#include"ChooseChannel_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
ChooseChannel_Base::ChooseChannel_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
ChooseChannel_Base::~ChooseChannel_Base()
{
}
