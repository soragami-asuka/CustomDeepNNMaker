//======================================
// �����֐����C���[
//======================================
#include"stdafx.h"

#include"Activation_FUNC.hpp"

#include"Activation_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
Activation_Base::Activation_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, ::CreateLearningSetting(), i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
Activation_Base::~Activation_Base()
{
}


//===========================
// ���C���[����
//===========================





//===========================
// �ŗL�֐�
//===========================
