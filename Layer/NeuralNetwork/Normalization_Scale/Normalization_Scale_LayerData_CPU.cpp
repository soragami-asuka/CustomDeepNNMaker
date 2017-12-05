//======================================
// �o�b�`���K���̃��C���[�f�[�^
// CPU����
//======================================
#include"stdafx.h"

#include"Normalization_Scale_LayerData_CPU.h"
#include"Normalization_Scale_FUNC.hpp"
#include"Normalization_Scale_CPU.h"

#include"Library/NeuralNetwork/Optimizer.h"


using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// �R���X�g���N�^ / �f�X�g���N�^
	//===========================
	/** �R���X�g���N�^ */
	Normalization_Scale_LayerData_CPU::Normalization_Scale_LayerData_CPU(const Gravisbell::GUID& guid)
		:	Normalization_Scale_LayerData_Base(guid)
	{
	}
	/** �f�X�g���N�^ */
	Normalization_Scale_LayerData_CPU::~Normalization_Scale_LayerData_CPU()
	{
	}


	//===========================
	// ������
	//===========================
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Normalization_Scale_LayerData_CPU::Initialize(void)
	{
		this->scale = 1.0f;
		this->bias  = 0.0f;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@param	i_config			�ݒ���
		@oaram	i_inputDataStruct	���̓f�[�^�\�����
		@return	���������ꍇ0 */
	ErrorCode Normalization_Scale_LayerData_CPU::Initialize(const SettingData::Standard::IData& i_data)
	{
		ErrorCode err;

		// �ݒ���̓o�^
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// ������
		err = this->Initialize();
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// �I�v�e�B�}�C�U�[�̐ݒ�
		err = this->ChangeOptimizer(L"SGD");
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �o�b�t�@����f�[�^��ǂݍ���
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@return	���������ꍇ0 */
	ErrorCode Normalization_Scale_LayerData_CPU::InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize)
	{
		int readBufferByte = 0;

		// �ݒ���
		S32 useBufferByte = 0;
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, useBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		readBufferByte += useBufferByte;
		this->SetLayerConfig(*pLayerStructure);
		delete pLayerStructure;

		// ����������
		this->Initialize();

		// �X�P�[�����O�l
		memcpy(&this->scale, &i_lpBuffer[readBufferByte], sizeof(this->scale));
		readBufferByte += sizeof(this->scale);
		// �o�C�A�X�l
		memcpy(&this->bias, &i_lpBuffer[readBufferByte], sizeof(this->bias));
		readBufferByte += sizeof(this->bias);


		// �I�v�e�B�}�C�U
		S32 useBufferSize = 0;
		// bias
		if(this->m_pOptimizer_bias)
			delete this->m_pOptimizer_bias;
		this->m_pOptimizer_bias = CreateOptimizerFromBuffer_CPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
		readBufferByte += useBufferSize;
		// neuron
		if(this->m_pOptimizer_scale)
			delete this->m_pOptimizer_scale;
		this->m_pOptimizer_scale = CreateOptimizerFromBuffer_CPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
		readBufferByte += useBufferSize;

		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 Normalization_Scale_LayerData_CPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// �ݒ���
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);


		// �X�P�[�����O�l
		memcpy(&o_lpBuffer[writeBufferByte], &this->scale, sizeof(this->scale));
		writeBufferByte += sizeof(this->scale);
		// �o�C�A�X�l
		memcpy(&o_lpBuffer[writeBufferByte], &this->bias, sizeof(this->bias));
		writeBufferByte += sizeof(this->bias);


		// �I�v�e�B�}�C�U
		// bias
		writeBufferByte += this->m_pOptimizer_bias->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
		// neuron
		writeBufferByte += this->m_pOptimizer_scale->WriteToBuffer(&o_lpBuffer[writeBufferByte]);


		return writeBufferByte;
	}


	//===========================
	// ���C���[�쐬
	//===========================
	/** ���C���[���쐬����.
		@param guid	�V�K�������郌�C���[��GUID. */
	ILayerBase* Normalization_Scale_LayerData_CPU::CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return NULL;

		return new Normalization_Scale_CPU(guid, *this, i_lpInputDataStruct[0]);
	}
	

	//===========================
	// �I�v�e�B�}�C�U�[�ݒ�
	//===========================		
	/** �I�v�e�B�}�C�U�[��ύX���� */
	ErrorCode Normalization_Scale_LayerData_CPU::ChangeOptimizer(const wchar_t i_optimizerID[])
	{
		ChangeOptimizer_CPU(&this->m_pOptimizer_bias,  i_optimizerID, 1);
		ChangeOptimizer_CPU(&this->m_pOptimizer_scale, i_optimizerID, 1);

		return ErrorCode::ERROR_CODE_NONE;
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;


/** Create a layer for CPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::Normalization_Scale_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::Normalization_Scale_LayerData_CPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// ������
	Gravisbell::ErrorCode errCode = pLayerData->Initialize(i_data);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	return pLayerData;
}
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::Normalization_Scale_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::Normalization_Scale_LayerData_CPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// ������
	S32 useBufferSize = 0;
	Gravisbell::ErrorCode errCode = pLayerData->InitializeFromBuffer(i_lpBuffer, i_bufferSize, useBufferSize);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	// �g�p�����o�b�t�@�ʂ��i�[
	o_useBufferSize = useBufferSize;

	return pLayerData;
}
