//======================================
// �����֐����C���[
//======================================
#include"stdafx.h"

#include"Activation_Discriminator_FUNC.hpp"

#include"Activation_Discriminator_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
Activation_Discriminator_Base::Activation_Discriminator_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	CNNSingle2SingleLayerBase(guid, ::CreateLearningSetting(), i_inputDataStruct, i_outputDataStruct)
{
}

/** �f�X�g���N�^ */
Activation_Discriminator_Base::~Activation_Discriminator_Base()
{
}


//===========================
// ���C���[����
//===========================


//===========================
// �ŗL�֐�
//===========================
