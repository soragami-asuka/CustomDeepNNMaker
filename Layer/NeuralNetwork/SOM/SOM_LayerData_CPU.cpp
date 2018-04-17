//======================================
// �S�����j���[�����l�b�g���[�N�̃��C���[�f�[�^
// CPU����
//======================================
#include"stdafx.h"

#include"SOM_LayerData_CPU.h"
#include"SOM_FUNC.hpp"
#include"SOM_CPU.h"

#include"../_LayerBase/CLayerBase_CPU.h"

#include"Library/NeuralNetwork/Optimizer.h"
#include"Library/NeuralNetwork/Initializer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// �R���X�g���N�^ / �f�X�g���N�^
	//===========================
	/** �R���X�g���N�^ */
	SOM_LayerData_CPU::SOM_LayerData_CPU(const Gravisbell::GUID& guid)
		:	SOM_LayerData_Base(guid)
	{
	}
	/** �f�X�g���N�^ */
	SOM_LayerData_CPU::~SOM_LayerData_CPU()
	{
	}


	//===========================
	// ������
	//===========================
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode SOM_LayerData_CPU::Initialize(void)
	{
		// ���̓o�b�t�@�����m�F
		unsigned int inputBufferCount = this->GetInputBufferCount();
		if(inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// ���j�b�g�����m�F
		unsigned int unitCount = this->GetUnitCount();
		if(unitCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// �o�b�t�@���m�ۂ��A�����l��ݒ�
		U32 inputCount  = inputBufferCount;
		U32 outputCount = this->layerStructure.DimensionCount;

		this->lpUnitData.resize(unitCount * inputBufferCount);
		this->lppUnitData.resize(unitCount);
		for(U32 unitNum=0; unitNum<this->lppUnitData.size(); unitNum++)
		{
			this->lppUnitData[unitNum] = &this->lpUnitData[unitNum * inputBufferCount];
		}

		// ���j�b�g������
		for(unsigned int bufNum=0; bufNum<this->lpUnitData.size(); bufNum++)
		{
			this->lpUnitData[bufNum] = Gravisbell::Layer::NeuralNetwork::GetInitializerManager().GetRandomValue(this->layerStructure.InitializeMinValue, this->layerStructure.InitializeMaxValue);
		}

		// �w�K�񐔏�����
		this->learnTime = 0;			/**< �w�K���s�� */

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@param	i_config			�ݒ���
		@oaram	i_inputDataStruct	���̓f�[�^�\�����
		@return	���������ꍇ0 */
	ErrorCode SOM_LayerData_CPU::Initialize(const SettingData::Standard::IData& i_data)
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

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �o�b�t�@����f�[�^��ǂݍ���
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@return	���������ꍇ0 */
	ErrorCode SOM_LayerData_CPU::InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize, S64& o_useBufferSize)
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

		// ���j�b�g
		memcpy(&this->lpUnitData[0], &i_lpBuffer[readBufferByte], this->lpUnitData.size() * sizeof(NEURON_TYPE));
		readBufferByte += (int)this->lpUnitData.size() * sizeof(F32);

		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	//==================================
	// SOM�֘A����
	//==================================
	/** �}�b�v�̃o�b�t�@���擾����.
		@param	o_lpMapBuffer	�}�b�v���i�[����z�X�g�������o�b�t�@. GetMapSize()�̖߂�l�̗v�f�����K�v. */
	Gravisbell::ErrorCode SOM_LayerData_CPU::GetMapBuffer(F32* o_lpMapBuffer)const
	{
		memcpy(o_lpMapBuffer, &this->lpUnitData[0], sizeof(F32)*this->GetMapSize());

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S64 SOM_LayerData_CPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		S64 writeBufferByte = 0;

		// �ݒ���
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		// ���j�b�g
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpUnitData[0], this->lpUnitData.size() * sizeof(F32));
		writeBufferByte += (int)this->lpUnitData.size() * sizeof(NEURON_TYPE);


		return writeBufferByte;
	}


	//===========================
	// ���C���[�쐬
	//===========================
	/** ���C���[���쐬����.
		@param guid	�V�K�������郌�C���[��GUID. */
	ILayerBase* SOM_LayerData_CPU::CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return NULL;

		return new CNNSingle2SingleLayerBase_CPU<SOM_CPU, SOM_LayerData_CPU>(guid, *this, i_lpInputDataStruct[0], i_temporaryMemoryManager);
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
	Gravisbell::Layer::NeuralNetwork::SOM_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::SOM_LayerData_CPU(guid);
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
	Gravisbell::Layer::NeuralNetwork::SOM_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::SOM_LayerData_CPU(guid);
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
