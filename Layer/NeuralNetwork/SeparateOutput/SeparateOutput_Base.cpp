//======================================
// �o�͐M���������C���[
//======================================
#include"stdafx.h"

#include"SeparateOutput_FUNC.hpp"

#include"SeparateOutput_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


/** �R���X�g���N�^ */
SeparateOutput_Base::SeparateOutput_Base(Gravisbell::GUID guid, const IODataStruct& i_inputDataStruct, const IODataStruct& i_outputDataStruct)
	:	guid				(guid)
	,	inputDataStruct		(i_inputDataStruct)
	,	outputDataStruct	(i_outputDataStruct)
	,	pLearnData			(NULL)
{
}

/** �f�X�g���N�^ */
SeparateOutput_Base::~SeparateOutput_Base()
{
	if(pLearnData != NULL)
		delete pLearnData;
}


//===========================
// ���C���[����
//===========================

/** ���C���[��ʂ̎擾.
	ELayerKind �̑g�ݍ��킹. */
U32 SeparateOutput_Base::GetLayerKindBase(void)const
{
	return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_MULT_OUTPUT;
}

/** ���C���[�ŗL��GUID���擾���� */
Gravisbell::GUID SeparateOutput_Base::GetGUID(void)const
{
	return this->guid;
}

/** ���C���[���ʃR�[�h���擾����.
	@param o_layerCode	�i�[��o�b�t�@
	@return ���������ꍇ0 */
Gravisbell::GUID SeparateOutput_Base::GetLayerCode(void)const
{
	return this->GetLayerData().GetLayerCode();
}

/** �o�b�`�T�C�Y���擾����.
	@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
unsigned int SeparateOutput_Base::GetBatchSize()const
{
	return this->batchSize;
}

/** ���C���[�̐ݒ�����擾���� */
const SettingData::Standard::IData* SeparateOutput_Base::GetLayerStructure()const
{
	return this->GetLayerData().GetLayerStructure();
}


//===========================
// ���̓��C���[�֘A
//===========================
/** ���̓f�[�^�\�����擾����.
	@return	���̓f�[�^�\�� */
IODataStruct SeparateOutput_Base::GetInputDataStruct()const
{
	return this->inputDataStruct;
}

/** ���̓o�b�t�@�����擾����. */
unsigned int SeparateOutput_Base::GetInputBufferCount()const
{
	return this->GetInputDataStruct().GetDataCount();
}


//===========================
// �o�̓��C���[�֘A
//===========================
/** �o�̓f�[�^�̏o�͐惌�C���[��. */
U32 SeparateOutput_Base::GetOutputToLayerCount()const
{
	return this->GetLayerData().GetOutputToLayerCount();
}
/** �o�̓f�[�^�\�����擾���� */
IODataStruct SeparateOutput_Base::GetOutputDataStruct()const
{
	return this->outputDataStruct;
}

/** �o�̓o�b�t�@�����擾���� */
unsigned int SeparateOutput_Base::GetOutputBufferCount()const
{
	return this->GetOutputDataStruct().GetDataCount();
}


//===========================
// �ŗL�֐�
//===========================
