//======================================
// �����֐����C���[
//======================================
#include"stdafx.h"

#include"Residual_FUNC.hpp"

#include"Residual_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
Residual_Base::Residual_Base(Gravisbell::GUID guid, const std::vector<IODataStruct>& i_lpInputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNMult2SingleLayerBase(guid, ::CreateLearningSetting(), i_lpInputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
Residual_Base::~Residual_Base()
{
}

