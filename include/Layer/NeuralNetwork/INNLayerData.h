//=======================================
// �j���[�����l�b�g���[�N�̃��C���[�Ɋւ���f�[�^����舵���C���^�[�t�F�[�X
// �o�b�t�@�Ȃǂ��Ǘ�����.
//=======================================
#ifndef __GRAVISBELL_I_NN_LAYER_DATA_H__
#define __GRAVISBELL_I_NN_LAYER_DATA_H__

#include"../ILayerData.h"

#include"./INNLayer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class INNLayerData : public ILayerData
	{
	public:
		/** �R���X�g���N�^ */
		INNLayerData() : ILayerData(){}
		/** �f�X�g���N�^ */
		virtual ~INNLayerData(){}

		
		//===========================
		// ���C���[�ݒ�
		//===========================
	public:
		/** ���C���[�̐ݒ�����擾���� */
		virtual const SettingData::Standard::IData* GetLayerStructure()const = 0;


		//===========================
		// ���̓��C���[�֘A
		//===========================
	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		virtual IODataStruct GetInputDataStruct()const = 0;

		/** ���̓o�b�t�@�����擾����. */
		virtual U32 GetInputBufferCount()const = 0;


		//===========================
		// �o�̓��C���[�֘A
		//===========================
	public:
		/** �o�̓f�[�^�\�����擾���� */
		virtual IODataStruct GetOutputDataStruct()const = 0;

		/** �o�̓o�b�t�@�����擾���� */
		virtual unsigned int GetOutputBufferCount()const = 0;


	public:
		//===========================
		// ���C���[�쐬
		//===========================
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		virtual INNLayer* CreateLayer(const Gravisbell::GUID& guid) = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif