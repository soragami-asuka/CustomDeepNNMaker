//======================================
// �����փ��C���[
//======================================
#ifndef __BatchNormalization_BASE_H__
#define __BatchNormalization_BASE_H__

#include<Layer/NeuralNetwork/INNLayer.h>

#include<vector>

#include"BatchNormalization_DATA.hpp"

#include"BatchNormalization_LayerData_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class BatchNormalization_Base : public Gravisbell::Layer::NeuralNetwork::INNLayer
	{
	protected:
		Gravisbell::GUID guid;	/**< ���C���[���ʗp��GUID */

		SettingData::Standard::IData* pLearnData;	/**< �w�K�ݒ���`�����R���t�B�O�N���X */

		unsigned int batchSize;	/**< �o�b�`�T�C�Y */

	public:
		/** �R���X�g���N�^ */
		BatchNormalization_Base(Gravisbell::GUID guid);

		/** �f�X�g���N�^ */
		virtual ~BatchNormalization_Base();

		//===========================
		// ���C���[����
		//===========================
	public:
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		U32 GetLayerKindBase(void)const;

		/** ���C���[�ŗL��GUID���擾���� */
		Gravisbell::GUID GetGUID(void)const;

		/** ���C���[�̎�ގ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		Gravisbell::GUID GetLayerCode(void)const;

		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		unsigned int GetBatchSize()const;


		//===========================
		// ���C���[�ݒ�
		//===========================
	public:
		/** ���C���[�̐ݒ�����擾���� */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// ���C���[�ۑ�
		//===========================
	public:
		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		unsigned int GetUseBufferByteCount()const;


		//===========================
		// ���̓��C���[�֘A
		//===========================
	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		virtual IODataStruct GetInputDataStruct()const;

		/** ���̓o�b�t�@�����擾����. */
		unsigned int GetInputBufferCount()const;


		//===========================
		// �o�̓��C���[�֘A
		//===========================
	public:
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const;

		/** �o�̓o�b�t�@�����擾���� */
		unsigned int GetOutputBufferCount()const;


		//===========================
		// �ŗL�֐�
		//===========================
	public:


		//===========================
		// ���C���[�f�[�^�֘A
		//===========================
	public:
		/** ���C���[�f�[�^���擾���� */
		virtual BatchNormalization_LayerData_Base& GetLayerData() = 0;
		virtual const BatchNormalization_LayerData_Base& GetLayerData()const = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
