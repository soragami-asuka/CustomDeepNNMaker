//======================================
// �v�[�����O���C���[�̃f�[�^
// CPU����
//======================================
#include"stdafx.h"

#include"MergeAdd_LayerData_CPU.h"
#include"MergeAdd_FUNC.hpp"
#include"MergeAdd_CPU.h"

#include"../_LayerBase/CLayerBase_CPU.h"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// �R���X�g���N�^ / �f�X�g���N�^
	//===========================
	/** �R���X�g���N�^ */
	MergeAdd_LayerData_CPU::MergeAdd_LayerData_CPU(const Gravisbell::GUID& guid)
		:	MergeAdd_LayerData_Base(guid)
	{
	}
	/** �f�X�g���N�^ */
	MergeAdd_LayerData_CPU::~MergeAdd_LayerData_CPU()
	{
	}


	//===========================
	// ���C���[�쐬
	//===========================
	/** ���C���[���쐬����.
		@param guid	�V�K�������郌�C���[��GUID. */
	ILayerBase* MergeAdd_LayerData_CPU::CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return NULL;

		std::vector<IODataStruct> lpInputDataStruct;
		for(U32 i=0; i<i_inputLayerCount; i++)
		{
			lpInputDataStruct.push_back(i_lpInputDataStruct[i]);
		}

		return new CNNMult2SingleLayerBase_CPU<MergeAdd_CPU,MergeAdd_LayerData_CPU>(guid, *this, lpInputDataStruct, i_temporaryMemoryManager);
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
	Gravisbell::Layer::NeuralNetwork::MergeAdd_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::MergeAdd_LayerData_CPU(guid);
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
	Gravisbell::Layer::NeuralNetwork::MergeAdd_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::MergeAdd_LayerData_CPU(guid);
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