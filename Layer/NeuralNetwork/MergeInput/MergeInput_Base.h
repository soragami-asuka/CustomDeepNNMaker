//======================================
// �����փ��C���[
//======================================
#ifndef __MergeInput_BASE_H__
#define __MergeInput_BASE_H__

#include<Layer/NeuralNetwork/INNMult2SingleLayer.h>

#include<vector>

#include"MergeInput_DATA.hpp"

#include"MergeInput_LayerData_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	typedef float NEURON_TYPE;	/**< �j���[�����Ɏg�p����f�[�^�^. float or double */

	class MergeInput_Base : public INNMult2SingleLayer
	{
	protected:
		Gravisbell::GUID guid;	/**< ���C���[���ʗp��GUID */

		SettingData::Standard::IData* pLearnData;	/**< �w�K�ݒ���`�����R���t�B�O�N���X */

		unsigned int batchSize;	/**< �o�b�`�T�C�Y */

	public:
		/** �R���X�g���N�^ */
		MergeInput_Base(Gravisbell::GUID guid);

		/** �f�X�g���N�^ */
		virtual ~MergeInput_Base();

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
		// ���̓��C���[�֘A
		//===========================
	public:
		/** ���̓f�[�^�̐����擾���� */
		U32 GetInputDataCount()const;

		/** ���̓f�[�^�\�����擾����.
			@return	���̓f�[�^�\�� */
		IODataStruct GetInputDataStruct(U32 i_dataNum)const;

		/** ���̓o�b�t�@�����擾����. */
		unsigned int GetInputBufferCount(U32 i_dataNum)const;


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
		virtual MergeInput_LayerData_Base& GetLayerData() = 0;
		virtual const MergeInput_LayerData_Base& GetLayerData()const = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif
