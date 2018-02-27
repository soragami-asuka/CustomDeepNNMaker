//======================================
// �S�����j���[�����l�b�g���[�N�̃��C���[�f�[�^
// GPU����
//======================================
#include"stdafx.h"

#include"SOM_LayerData_GPU.cuh"
#include"SOM_FUNC.hpp"
#include"SOM_GPU.cuh"

#include"../_LayerBase/CLayerBase_GPU.cuh"

#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)

#include"Library/NeuralNetwork/Optimizer.h"
#include"Library/NeuralNetwork/Initializer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// �R���X�g���N�^ / �f�X�g���N�^
	//===========================
	/** �R���X�g���N�^ */
	SOM_LayerData_GPU::SOM_LayerData_GPU(const Gravisbell::GUID& guid)
		:	SOM_LayerData_Base(guid)
	{
	}
	/** �f�X�g���N�^ */
	SOM_LayerData_GPU::~SOM_LayerData_GPU()
	{
	}


	//===========================
	// ������
	//===========================
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode SOM_LayerData_GPU::Initialize(void)
	{
		// ���̓o�b�t�@�����m�F
		unsigned int inputBufferCount = this->GetInputBufferCount();
		if(inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// �j���[���������m�F
		unsigned int unitCount = this->GetUnitCount();
		if(unitCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// �o�b�t�@���m�ۂ��A�����l��ݒ�
		U32 inputCount  = inputBufferCount;
		U32 outputCount = this->layerStructure.DimensionCount;

		this->lpUnitData.resize(unitCount * inputBufferCount);

		thrust::host_vector<F32> lpTmpUnitData(unitCount * inputBufferCount);

		for(U32 i=0; i<lpTmpUnitData.size(); i++)
		{
			lpTmpUnitData[i] = Gravisbell::Layer::NeuralNetwork::GetInitializerManager().GetRandomValue(this->layerStructure.InitializeMinValue, this->layerStructure.InitializeMaxValue);
		}

		this->lpUnitData = lpTmpUnitData;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@param	i_config			�ݒ���
		@oaram	i_inputDataStruct	���̓f�[�^�\�����
		@return	���������ꍇ0 */
	ErrorCode SOM_LayerData_GPU::Initialize(const SettingData::Standard::IData& i_data)
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

		return this->Initialize();
	}
	/** ������. �o�b�t�@����f�[�^��ǂݍ���
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@return	���������ꍇ0 */
	ErrorCode SOM_LayerData_GPU::InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize, S64& o_useBufferSize )
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

		// �o�b�t�@����R�s�[
		// �j���[����
		cudaMemcpy(
			thrust::raw_pointer_cast(&this->lpUnitData[0]),
			&i_lpBuffer[readBufferByte],
			sizeof(F32) * this->lpUnitData.size(),
			cudaMemcpyHostToDevice);
		readBufferByte += sizeof(F32) * (S32)this->lpUnitData.size();


		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S64 SOM_LayerData_GPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		S64 writeBufferByte = 0;

		// �ݒ���
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		// �j���[����
		cudaMemcpy(
			&o_lpBuffer[writeBufferByte],
			thrust::raw_pointer_cast(&this->lpUnitData[0]),
			sizeof(F32) * this->lpUnitData.size(),
			cudaMemcpyDeviceToHost);
		writeBufferByte += sizeof(F32) * (S32)this->lpUnitData.size();


		return writeBufferByte;
	}


	//===========================
	// ���C���[�쐬
	//===========================
	/** ���C���[���쐬����.
		@param guid	�V�K�������郌�C���[��GUID. */
	ILayerBase* SOM_LayerData_GPU::CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return NULL;

		return new CNNSingle2SingleLayerBase_GPU<SOM_GPU, SOM_LayerData_GPU>(guid, *this, i_lpInputDataStruct[0], i_temporaryMemoryManager);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;


using namespace Gravisbell;

/** Create a layer for GPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::SOM_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::SOM_LayerData_GPU(guid);
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
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::SOM_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::SOM_LayerData_GPU(guid);
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