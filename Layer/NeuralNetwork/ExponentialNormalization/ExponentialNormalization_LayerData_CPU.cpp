//======================================
// �o�b�`���K���̃��C���[�f�[�^
// CPU����
//======================================
#include"stdafx.h"

#include"ExponentialNormalization_LayerData_CPU.h"
#include"ExponentialNormalization_FUNC.hpp"
#include"ExponentialNormalization_CPU.h"

#include"Library/NeuralNetwork/Optimizer.h"

#include"../_LayerBase/CLayerBase_CPU.h"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// �R���X�g���N�^ / �f�X�g���N�^
	//===========================
	/** �R���X�g���N�^ */
	ExponentialNormalization_LayerData_CPU::ExponentialNormalization_LayerData_CPU(const Gravisbell::GUID& guid)
		:	ExponentialNormalization_LayerData_Base(guid)
	{
	}
	/** �f�X�g���N�^ */
	ExponentialNormalization_LayerData_CPU::~ExponentialNormalization_LayerData_CPU()
	{
	}


	//===========================
	// ������
	//===========================
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode ExponentialNormalization_LayerData_CPU::Initialize(void)
	{
		this->lpMean.resize(this->layerStructure.InputChannelCount);
		this->lpVariance.resize(this->layerStructure.InputChannelCount);

		for(U32 ch=0; ch<this->layerStructure.InputChannelCount; ch++)
		{
			this->lpMean[ch] = 0.0f;
			this->lpVariance[ch] = 1.0f;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@param	i_config			�ݒ���
		@oaram	i_inputDataStruct	���̓f�[�^�\�����
		@return	���������ꍇ0 */
	ErrorCode ExponentialNormalization_LayerData_CPU::Initialize(const SettingData::Standard::IData& i_data)
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
	ErrorCode ExponentialNormalization_LayerData_CPU::InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize, S64& o_useBufferSize)
	{
		S64 readBufferByte = 0;

		// �ݒ���
		S64 useBufferByte = 0;
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, useBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		readBufferByte += useBufferByte;
		this->SetLayerConfig(*pLayerStructure);
		delete pLayerStructure;

		// ����������
		this->Initialize();

		// ����
		memcpy(&this->lpMean[0], &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size());
		readBufferByte += sizeof(F32)*(S32)this->lpMean.size();
		// ���U
		memcpy(&this->lpVariance[0], &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size());
		readBufferByte += sizeof(F32)*(S32)this->lpMean.size();


		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S64 ExponentialNormalization_LayerData_CPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		S64 writeBufferByte = 0;

		// �ݒ���
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		// ����
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpMean[0], sizeof(F32)*this->lpMean.size());
		writeBufferByte += sizeof(F32)*(S32)this->lpMean.size();
		// ���U
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpVariance[0], sizeof(F32)*this->lpVariance.size());
		writeBufferByte += sizeof(F32)*(S32)this->lpVariance.size();


		return writeBufferByte;
	}


	//===========================
	// ���C���[�쐬
	//===========================
	/** ���C���[���쐬����.
		@param guid	�V�K�������郌�C���[��GUID. */
	ILayerBase* ExponentialNormalization_LayerData_CPU::CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return NULL;

		return new CNNSingle2SingleLayerBase_CPU<ExponentialNormalization_CPU, ExponentialNormalization_LayerData_CPU>(guid, *this, i_lpInputDataStruct[0], i_temporaryMemoryManager);
	}


	//===========================
	// �I�v�e�B�}�C�U�[�ݒ�
	//===========================		
	/** �I�v�e�B�}�C�U�[��ύX���� */
	ErrorCode ExponentialNormalization_LayerData_CPU::ChangeOptimizer(const wchar_t i_optimizerID[])
	{
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
	Gravisbell::Layer::NeuralNetwork::ExponentialNormalization_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::ExponentialNormalization_LayerData_CPU(guid);
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
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::ExponentialNormalization_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::ExponentialNormalization_LayerData_CPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// ������
	S64 useBufferSize = 0;
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