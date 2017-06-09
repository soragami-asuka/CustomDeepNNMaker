//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A����������������
//======================================
#ifndef __UpSampling_BASE_H__
#define __UpSampling_BASE_H__

#include<Layer/NeuralNetwork/INNSingle2SingleLayer.h>

#include<vector>

#include"UpSampling_DATA.hpp"

#include"UpSampling_LayerData_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class UpSampling_Base : public INNSingle2SingleLayer
	{
	protected:
		Gravisbell::GUID guid;	/**< ���C���[���ʗp��GUID */

		SettingData::Standard::IData* pLearnData;		/**< �w�K�ݒ���`�����R���t�B�O�N���X */
//		UpSampling::LearnDataStructure learnData;	/**< �w�K�ݒ� */

		U32 batchSize;	/**< �o�b�`�T�C�Y */

	public:
		/** �R���X�g���N�^ */
		UpSampling_Base(Gravisbell::GUID guid);

		/** �f�X�g���N�^ */
		virtual ~UpSampling_Base();

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
		U32 GetBatchSize()const;


		//===========================
		// ���C���[�ݒ�
		//===========================
	public:
		/** ���C���[�̐ݒ�����擾���� */
		const SettingData::Standard::IData* GetLayerStructure()const;


		//===========================
		// ���̓��C���[�֘A
		//===========================
	public:
		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		virtual IODataStruct GetInputDataStruct()const;

		/** ���̓o�b�t�@�����擾����. */
		U32 GetInputBufferCount()const;


		//===========================
		// �o�̓��C���[�֘A
		//===========================
	public:
		/** �o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct()const;

		/** �o�̓o�b�t�@�����擾���� */
		U32 GetOutputBufferCount()const;


		//===========================
		// �ŗL�֐�
		//===========================
	public:


		//===========================
		// ���C���[�f�[�^�֘A
		//===========================
	public:
		/** ���C���[�f�[�^���擾���� */
		virtual UpSampling_LayerData_Base& GetLayerData() = 0;
		virtual const UpSampling_LayerData_Base& GetLayerData()const = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
