//======================================
// 出力信号分割レイヤー
//======================================
#include"stdafx.h"

#include"SignalArray2Value_FUNC.hpp"

#include"SignalArray2Value_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** コンストラクタ */
SignalArray2Value_Base::SignalArray2Value_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, i_inputDataStruct, i_outputDataStruct)
{
}

/** デストラクタ */
SignalArray2Value_Base::~SignalArray2Value_Base()
{
}
