//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̃��C���[�f�[�^
// GPU����
//======================================
#include"stdafx.h"

#include"UpSampling_LayerData_GPU.cuh"
#include"UpSampling_FUNC.hpp"
#include"UpSampling_GPU.cuh"

#include"../_LayerBase/CLayerBase_GPU.cuh"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// �R���X�g���N�^ / �f�X�g���N�^
	//===========================
	/** �R���X�g���N�^ */
	UpSampling_LayerData_GPU::UpSampling_LayerData_GPU(const Gravisbell::GUID& guid)
		:	UpSampling_LayerData_Base(guid)
	{
	}
	/** �f�X�g���N�^ */
	UpSampling_LayerData_GPU::~UpSampling_LayerData_GPU()
	{
	}


	//===========================
	// ���C���[�쐬
	//===========================
	/** ���C���[���쐬����.
		@param guid	�V�K�������郌�C���[��GUID. */
	ILayerBase* UpSampling_LayerData_GPU::CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return NULL;

		return new CNNSingle2SingleLayerBase_GPU<UpSampling_GPU,UpSampling_LayerData_GPU>(guid, *this, i_lpInputDataStruct[0], i_temporaryMemoryManager);
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
	Gravisbell::Layer::NeuralNetwork::UpSampling_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::UpSampling_LayerData_GPU(guid);
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
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::UpSampling_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::UpSampling_LayerData_GPU(guid);
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
