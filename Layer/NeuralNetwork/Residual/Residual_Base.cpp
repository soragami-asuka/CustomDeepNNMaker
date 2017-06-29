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
	:	guid				(guid)
	,	lpInputDataStruct	(i_lpInputDataStruct)
	,	outputDataStruct	(i_outputDataStruct)
	,	pLearnData			(NULL)
{
}

/** �f�X�g���N�^ */
Residual_Base::~Residual_Base()
{
	if(pLearnData != NULL)
		delete pLearnData;
}


//===========================
// ���C���[����
//===========================

/** ���C���[��ʂ̎擾.
	ELayerKind �̑g�ݍ��킹. */
U32 Residual_Base::GetLayerKindBase(void)const
{
	return Gravisbell::Layer::ELayerKind::LAYER_KIND_NEURALNETWORK | Gravisbell::Layer::ELayerKind::LAYER_KIND_MULT_INPUT | Gravisbell::Layer::ELayerKind::LAYER_KIND_SINGLE_OUTPUT;
}

/** ���C���[�ŗL��GUID���擾���� */
Gravisbell::GUID Residual_Base::GetGUID(void)const
{
	return this->guid;
}

/** ���C���[���ʃR�[�h���擾����.
	@param o_layerCode	�i�[��o�b�t�@
	@return ���������ꍇ0 */
Gravisbell::GUID Residual_Base::GetLayerCode(void)const
{
	return this->GetLayerData().GetLayerCode();
}

/** �o�b�`�T�C�Y���擾����.
	@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
unsigned int Residual_Base::GetBatchSize()const
{
	return this->batchSize;
}

/** ���C���[�̐ݒ�����擾���� */
const SettingData::Standard::IData* Residual_Base::GetLayerStructure()const
{
	return this->GetLayerData().GetLayerStructure();
}


//===========================
// ���̓��C���[�֘A
//===========================
/** ���̓f�[�^�̐����擾���� */
U32 Residual_Base::GetInputDataCount()const
{
	return (U32)this->lpInputDataStruct.size();
}

/** ���̓f�[�^�\�����擾����.
	@return	���̓f�[�^�\�� */
IODataStruct Residual_Base::GetInputDataStruct(U32 i_dataNum)const
{
	if(i_dataNum >= this->GetInputDataCount())
		return IODataStruct(0,0,0,0);

	return this->lpInputDataStruct[i_dataNum];
}

/** ���̓o�b�t�@�����擾����. */
unsigned int Residual_Base::GetInputBufferCount(U32 i_dataNum)const
{
	return this->GetInputDataStruct(i_dataNum).GetDataCount();
}


//===========================
// �o�̓��C���[�֘A
//===========================
/** �o�̓f�[�^�\�����擾���� */
IODataStruct Residual_Base::GetOutputDataStruct()const
{
	return this->outputDataStruct;
}

/** �o�̓o�b�t�@�����擾���� */
unsigned int Residual_Base::GetOutputBufferCount()const
{
	return this->GetOutputDataStruct().GetDataCount();
}


//===========================
// �ŗL�֐�
//===========================
