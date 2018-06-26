//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̃��C���[�f�[�^
// CPU����
//======================================
#include"stdafx.h"

#include"Convolution_LayerData_CPU.h"
#include"Convolution_FUNC.hpp"
#include"Convolution_CPU.h"

#include"Library/NeuralNetwork/Optimizer.h"
#include"Library/NeuralNetwork/Initializer.h"
#include<Library/NeuralNetwork/WeightData.h>

#include"../_LayerBase/CLayerBase_CPU.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// �R���X�g���N�^ / �f�X�g���N�^
	//===========================
	/** �R���X�g���N�^ */
	Convolution_LayerData_CPU::Convolution_LayerData_CPU(const Gravisbell::GUID& guid)
		:	Convolution_LayerData_Base(guid)
	{
	}
	/** �f�X�g���N�^ */
	Convolution_LayerData_CPU::~Convolution_LayerData_CPU()
	{
	}


	//===========================
	// ������
	//===========================
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Convolution_LayerData_CPU::Initialize(void)
	{
		// ���̓o�b�t�@�����m�F
		U32 inputBufferCount = this->layerStructure.Input_Channel * this->layerStructure.FilterSize.z * this->layerStructure.FilterSize.y * this->layerStructure.FilterSize.x;
		if(inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// �j���[���������m�F
		U32 neuronCount = this->layerStructure.Output_Channel;
		if(neuronCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// �o�b�t�@���m�ۂ��A�����l��ݒ�
		U32 inputCount  = inputBufferCount;
		U32 outputCount = neuronCount;
		if(this->pWeightData)
			delete this->pWeightData;
		this->pWeightData = Gravisbell::Layer::NeuralNetwork::GetWeightDataManager().CreateWeightData_CPU(this->layerStructure.WeightData, neuronCount, inputBufferCount);
		if(this->pWeightData == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;
		this->pWeightData->Initialize(this->layerStructure.Initializer, inputCount, outputCount);

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================


	//===========================
	// ���C���[�쐬
	//===========================
	/** ���C���[���쐬����.
		@param guid	�V�K�������郌�C���[��GUID. */
	ILayerBase* Convolution_LayerData_CPU::CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return NULL;

		return new CNNSingle2SingleLayerBase_CPU<Convolution_CPU, Convolution_LayerData_CPU>(guid, *this, i_lpInputDataStruct[0], i_temporaryMemoryManager);
	}


	//===========================
	// �I�v�e�B�}�C�U�[�ݒ�
	//===========================

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

using namespace Gravisbell;

/** Create a layer for CPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::Convolution_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::Convolution_LayerData_CPU(guid);
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
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize)
{
	// �쐬
	Gravisbell::Layer::NeuralNetwork::Convolution_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::Convolution_LayerData_CPU(guid);
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
