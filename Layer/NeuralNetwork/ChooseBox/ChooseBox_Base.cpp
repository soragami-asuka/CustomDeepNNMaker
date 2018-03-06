//======================================
// 出力信号分割レイヤー
//======================================
#include"stdafx.h"

#include"ChooseBox_FUNC.hpp"

#include"ChooseBox_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
ChooseBox_Base::ChooseBox_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
ChooseBox_Base::~ChooseBox_Base()
{
}
